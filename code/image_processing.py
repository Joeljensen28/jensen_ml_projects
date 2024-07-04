from PIL import Image
import torchvision.transforms as transforms

def image_to_tensor(image_file):
    # Parameter is the path to the image file. PIL library opens the image and converts to grayscale
    image = Image.open(image_file).convert('L')

    transform = transforms.Compose([
        # Resize image to 28x28 so it can fit into the net
        transforms.Resize((28, 28)),
        # Converts the image to tensor
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Transforms image to fit the net
    image = transform(image)
    
    image = image.unsqueeze(0)
    return image