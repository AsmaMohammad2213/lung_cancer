import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
import uuid

# ----------------------------------------------------------
# 1️⃣ Flask Setup
# ----------------------------------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ----------------------------------------------------------
# 2️⃣ Model Definition (same as your training model)
# ----------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class PrimaryCapsule(nn.Module):
    def __init__(self, in_channels, num_capsules, capsule_dim, kernel_size, stride):
        super().__init__()
        self.capsules = nn.Conv2d(in_channels, num_capsules * capsule_dim, kernel_size, stride)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim

    def forward(self, x):
        out = self.capsules(x)
        N, C, H, W = out.size()
        out = out.view(N, self.num_capsules, self.capsule_dim, H, W)
        out = out.view(N, self.num_capsules, -1)
        return self.squash(out)

    def squash(self, s):
        mag_sq = torch.sum(s ** 2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq + 1e-9)
        return (mag_sq / (1.0 + mag_sq)) * (s / mag)


class DigitCapsule(nn.Module):
    def __init__(self, num_caps_in, dim_caps_in, num_caps_out, dim_caps_out, num_routes):
        super().__init__()
        self.num_routes = num_routes
        self.num_caps_out = num_caps_out
        self.W = nn.Parameter(torch.randn(1, num_routes, num_caps_out, dim_caps_out, dim_caps_in))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(2).unsqueeze(4)
        u_hat = torch.matmul(self.W, x)
        b_ij = torch.zeros(1, self.num_routes, self.num_caps_out, 1, 1, device=x.device)
        for iteration in range(3):
            c_ij = torch.softmax(b_ij, dim=1)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            if iteration < 2:
                b_ij = b_ij + (u_hat * v_j).sum(dim=-2, keepdim=True)
        return v_j.squeeze(1)

    def squash(self, s):
        mag_sq = torch.sum(s ** 2, dim=-1, keepdim=True)
        mag = torch.sqrt(mag_sq + 1e-9)
        return (mag_sq / (1.0 + mag_sq)) * (s / mag)


class RS_CapsNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.resblock1 = ResidualBlock(64, 128, stride=2)
        self.resblock2 = ResidualBlock(128, 256, stride=2)
        self.primary_caps = PrimaryCapsule(256, 8, 8, 3, 2)
        self.digit_caps = DigitCapsule(num_caps_in=8, dim_caps_in=1800, num_caps_out=num_classes, dim_caps_out=16, num_routes=8)
        self.decoder = nn.Sequential(
            nn.Linear(16 * num_classes, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.primary_caps(x)
        x = self.digit_caps(x)
        x = x.view(x.size(0), -1)
        return self.decoder(x)


# ----------------------------------------------------------
# 3️⃣ Load Model
# ----------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RS_CapsNet(num_classes=3).to(device)
model.load_state_dict(torch.load('models/RS_CapsNet.pth', map_location=device))
model.eval()

# ----------------------------------------------------------
# 4️⃣ Transform (same as training)
# ----------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----------------------------------------------------------
# 5️⃣ Class names (update with your own)
# ----------------------------------------------------------
class_names = ['Benign case', 'Malignant case', 'Normal case']  # Replace with your actual classes

# ----------------------------------------------------------
# 6️⃣ Routes
# ----------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    filename = f"{uuid.uuid4().hex}_{file.filename}"  # ✅ define before print
    print(f"Uploaded file: {filename}")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image = Image.open(filepath).convert('L')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        predicted_idx = probs.argmax()
        predicted_class = class_names[predicted_idx]
        confidence = probs[predicted_idx] * 100

    return render_template('result.html',
                           filename=filename,
                           predicted_class=predicted_class,
                           confidence=confidence,
                           probs=zip(class_names, probs))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# ----------------------------------------------------------
# 8️⃣ Run
# ----------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
