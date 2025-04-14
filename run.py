import argparse
import os

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Facial Expression Recognition')
    parser.add_argument('--mode', type=str, default='train_test',
                      help='Mode: train_test, analyze_image, or webcam')
    parser.add_argument('--image', type=str, default=None,
                      help='Path to image file for analysis (only used with --mode analyze_image)')
    args = parser.parse_args()
    
    # Mode selection
    mode = args.mode.lower()
    
    if mode == 'train_test':
        # Import here to avoid circular imports
        from scripts.train_model import train_and_test
        train_and_test()
    elif mode == 'analyze_image':
        from scripts.analyze_image import analyze_image
        analyze_image(args.image)
    elif mode == 'webcam':
        from scripts.test_webcam import test_webcam
        from src.model.model_prediction import load_model
        model = load_model()
        test_webcam(model)
    elif mode == 'web':
        print('Starting UI...')
        from app import create_app
        app = create_app()
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print(f"Invalid mode: {mode}")
        print("Available modes: train_test, analyze_image, webcam")

if __name__ == "__main__":
    main()