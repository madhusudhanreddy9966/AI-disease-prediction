import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class ARVisualizer:
    def __init__(self):
        self.disease_info = {
            'Acne and Rosacea Photos': {
                'description': 'Inflammatory skin condition affecting hair follicles',
                'treatment': 'Topical retinoids, antibiotics, gentle cleansing',
                'color': (255, 100, 100)
            },
            'Atopic Dermatitis Photos': {
                'description': 'Chronic inflammatory skin condition (eczema)',
                'treatment': 'Moisturizers, topical corticosteroids, avoid triggers',
                'color': (100, 255, 100)
            },
            'Cellulitis Impetigo and other Bacterial Infections': {
                'description': 'Bacterial skin infection requiring medical attention',
                'treatment': 'Antibiotics, proper wound care, medical consultation',
                'color': (255, 100, 255)
            },
            'Eczema Photos': {
                'description': 'Inflammatory skin condition causing itchy, red patches',
                'treatment': 'Moisturizers, topical treatments, avoid irritants',
                'color': (100, 100, 255)
            },
            'Light Diseases and Disorders of Pigmentation': {
                'description': 'Skin pigmentation disorders',
                'treatment': 'Sun protection, topical treatments, dermatologist consultation',
                'color': (255, 255, 100)
            },
            'Psoriasis pictures Lichen Planus and related diseases': {
                'description': 'Autoimmune skin condition with scaly patches',
                'treatment': 'Topical treatments, phototherapy, systemic medications',
                'color': (100, 255, 255)
            },
            'Seborrheic Keratoses and other Benign Tumors': {
                'description': 'Benign skin growths, usually harmless',
                'treatment': 'Monitoring, removal if cosmetically desired',
                'color': (255, 150, 50)
            }
        }
    
    def create_ar_overlay(self, image, disease_name, confidence):
        """Create AR overlay with disease information"""
        if isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            img = image
        
        # Create overlay
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Get disease info
        info = self.disease_info.get(disease_name, {
            'description': 'Unknown condition',
            'treatment': 'Consult a dermatologist',
            'color': (128, 128, 128)
        })
        
        # Draw AR elements
        width, height = img.size
        
        # Main info box
        box_height = 150
        box_y = height - box_height - 20
        draw.rectangle([(20, box_y), (width-20, height-20)], 
                      fill=(*info['color'], 180), outline=info['color'], width=3)
        
        # Disease name
        try:
            font_large = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 16)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        draw.text((30, box_y + 10), f"Detected: {disease_name}", 
                 fill=(255, 255, 255), font=font_large)
        
        # Confidence
        draw.text((30, box_y + 40), f"Confidence: {confidence:.1%}", 
                 fill=(255, 255, 255), font=font_small)
        
        # Description
        draw.text((30, box_y + 65), f"Info: {info['description']}", 
                 fill=(255, 255, 255), font=font_small)
        
        # Treatment
        draw.text((30, box_y + 90), f"Treatment: {info['treatment']}", 
                 fill=(255, 255, 255), font=font_small)
        
        # Warning indicator
        if confidence < 0.7:
            draw.text((30, box_y + 115), "⚠️ Low confidence - Consult a doctor", 
                     fill=(255, 255, 0), font=font_small)
        
        # Combine images
        result = Image.alpha_composite(img.convert('RGBA'), overlay)
        return result.convert('RGB')
    
    def create_3d_marker(self, image, disease_name, confidence):
        """Create 3D-like marker for AR effect"""
        if isinstance(image, np.ndarray):
            img = image.copy()
        else:
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        height, width = img.shape[:2]
        
        # Get disease color
        info = self.disease_info.get(disease_name, {'color': (128, 128, 128)})
        color = info['color']
        
        # Draw 3D marker
        center = (width // 2, height // 2)
        
        # Outer circle (shadow effect)
        cv2.circle(img, center, 60, (0, 0, 0), 8)
        cv2.circle(img, (center[0]-2, center[1]-2), 60, color, 6)
        
        # Inner circle
        cv2.circle(img, center, 40, color, -1)
        cv2.circle(img, center, 40, (255, 255, 255), 3)
        
        # Confidence arc
        angle = int(360 * confidence)
        axes = (35, 35)
        cv2.ellipse(img, center, axes, 0, 0, angle, (0, 255, 0), 4)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{confidence:.0%}"
        text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + text_size[1] // 2
        cv2.putText(img, text, (text_x, text_y), font, 0.7, (255, 255, 255), 2)
        
        return img