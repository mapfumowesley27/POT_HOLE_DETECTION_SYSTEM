import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pothole_detector import PotholeDetector

def train_pothole_model(dataset_path, model_save_path=None, epochs=20, batch_size=32):
    """
    Train a pothole detection model using transfer learning.
    
    Expected dataset structure:
    dataset/
        train/
            no_pothole/
            small_pothole/
            medium_pothole/
            large_pothole/
        val/
            no_pothole/
            small_pothole/
            medium_pothole/
            large_pothole/
    """
    
    if model_save_path is None:
        model_save_path = os.path.join(os.path.dirname(__file__), 'models', 'pothole_model.h5')
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # 1. Initialize detector and create model structure
    detector = PotholeDetector()
    detector.create_model()
    model = detector.model
    
    # 2. Prepare Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    
    if not os.path.exists(train_dir):
        print(f"❌ Error: Training directory not found at {train_dir}")
        print("Please organize your dataset as follows:")
        print("dataset_path/train/[no_pothole, small_pothole, medium_pothole, large_pothole]")
        return

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=detector.input_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=detector.class_names
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=detector.input_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=detector.class_names
    ) if os.path.exists(val_dir) else None

    # 3. Phase 1: Train only the top layers
    print("🚀 Starting Phase 1: Training top layers...")
    model.fit(
        train_generator,
        epochs=epochs // 2,
        validation_data=val_generator,
        verbose=1
    )
    
    # 4. Phase 2: Fine-tuning (optional, unfreeze some layers of MobileNetV2)
    print("🚀 Starting Phase 2: Fine-tuning...")
    # Unfreeze the base model
    model.layers[0].trainable = True
    
    # Re-compile with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
        train_generator,
        epochs=epochs // 2,
        validation_data=val_generator,
        verbose=1
    )
    
    # 5. Save the final model
    model.save(model_save_path)
    print(f"✅ Model saved successfully to {model_save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Pothole Detection Model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--output', type=str, default=None, help='Path to save the model')
    
    args = parser.parse_args()
    train_pothole_model(args.dataset, args.output, args.epochs)
