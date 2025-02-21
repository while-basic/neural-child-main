# create_test_image.py
# Description: Script to create test images for vision development
# Created by: Christopher Celaya

from PIL import Image, ImageDraw
from pathlib import Path

def create_basic_shapes():
    """Create an image with basic shapes for early vision development"""
    # Create a white background
    img = Image.new('RGB', (300, 300), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw a black rectangle
    draw.rectangle([50, 50, 250, 250], outline='black', width=2)
    
    # Draw a red circle
    draw.ellipse([100, 100, 200, 200], fill='red')
    
    # Draw a blue triangle
    draw.polygon([(150, 75), (75, 225), (225, 225)], fill='blue')
    
    return img

def create_face_pattern():
    """Create a simple face pattern for social development"""
    img = Image.new('RGB', (300, 300), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw face circle
    draw.ellipse([50, 50, 250, 250], outline='black', width=2)
    
    # Draw eyes
    draw.ellipse([100, 100, 140, 140], fill='black')  # Left eye
    draw.ellipse([160, 100, 200, 140], fill='black')  # Right eye
    
    # Draw smile
    draw.arc([100, 120, 200, 200], 0, 180, fill='black', width=2)
    
    return img

def main():
    # Create directories if they don't exist
    development_path = Path('images/development')
    dreams_path = Path('images/dreams')
    development_path.mkdir(parents=True, exist_ok=True)
    dreams_path.mkdir(parents=True, exist_ok=True)
    
    # Create and save basic shapes image
    shapes_img = create_basic_shapes()
    shapes_img.save(development_path / 'basic_shapes.png')
    print("Created basic shapes image")
    
    # Create and save face pattern image
    face_img = create_face_pattern()
    face_img.save(development_path / 'face_pattern.png')
    print("Created face pattern image")

if __name__ == "__main__":
    main() 