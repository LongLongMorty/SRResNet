from PIL import Image
import torch
from torchvision.transforms import transforms
import SRResNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载训练后的模型参数
model = SRResNet.SRResNet()  # 创建模型实例
model.load_state_dict(torch.load('/home/featurize/pythonProject5/model.pth'))  # 加载模型参数
model = model.to(device)  # 将模型移动到设备
model.eval()  # 将模型设置为评估模式

# 加载要进行超分辨率处理的图像
image_path = '/home/featurize/pythonProject5/input/2.png'  # 新图像的路径
image = Image.open(image_path).convert('RGB')  # 加载图像
preprocess = transforms.ToTensor()  # 图像预处理
input_image = preprocess(image).unsqueeze(0).to(device)  # 转换为模型所需的张量格式并移动到设备

# 使用模型进行超分辨率处理
with torch.no_grad():
    output_image = model(input_image).squeeze().cpu()  # 使用模型进行推断并将输出移动到CPU

# 将输出图像保存到文件中
output_image = output_image.clamp(0, 1)  # 将输出图像的像素值限制在0到1之间
output_image = transforms.ToPILImage()(output_image)  # 将张量转换为PIL图像
output_image.save('/home/featurize/pythonProject5/output/output_image2.jpg')  # 保存输出图像