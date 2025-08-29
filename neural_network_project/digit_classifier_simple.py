#!/usr/bin/env python3
"""
Digit Classifier - Neural Network Project (Simplified Version)
A deep learning project for beginners using MNIST dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report
import os

def load_and_prepare_data():
    """Load MNIST dataset and prepare it for training"""
    print("📊 Loading MNIST dataset...")
    
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Training images: {x_train.shape[0]} (28x28 pixels)")
    print(f"   Testing images: {x_test.shape[0]} (28x28 pixels)")
    print(f"   Image size: {x_train.shape[1]}x{x_train.shape[2]} pixels")
    print(f"   Number of classes: {len(np.unique(y_train))}")
    
    # Normalize pixel values (0-255 -> 0-1)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Reshape for neural network (add channel dimension)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Convert labels to categorical (one-hot encoding)
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)
    
    print("✅ Data prepared for training!")
    print(f"   Training shape: {x_train.shape}")
    print(f"   Testing shape: {x_test.shape}")
    
    return x_train, x_test, y_train, y_test, y_train_cat, y_test_cat

def explore_dataset(x_train, y_train):
    """Show sample images from the dataset"""
    print("\n🖼️  Exploring the dataset...")
    
    # Create a grid of sample images
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle('Sample Handwritten Digits from MNIST Dataset', fontsize=16)
    
    for i in range(15):
        row = i // 5
        col = i % 5
        
        # Get random image
        idx = np.random.randint(0, len(x_train))
        img = x_train[idx].reshape(28, 28)
        label = y_train[idx]
        
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'Digit: {label}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_sample_digits.png', dpi=300, bbox_inches='tight')
    print("✅ Saved sample digits as 'mnist_sample_digits.png'")
    plt.show()

def build_neural_network():
    """Build a simple neural network for digit classification"""
    print("\n🏗️  Building neural network...")
    
    # Create the model
    model = keras.Sequential([
        # Input layer - flatten the 28x28x1 image
        layers.Flatten(input_shape=(28, 28, 1)),
        
        # Hidden layers
        layers.Dense(128, activation='relu', name='hidden_layer_1'),
        layers.Dropout(0.2),  # Prevent overfitting
        
        layers.Dense(64, activation='relu', name='hidden_layer_2'),
        layers.Dropout(0.2),
        
        # Output layer - 10 classes (digits 0-9)
        layers.Dense(10, activation='softmax', name='output_layer')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    print("✅ Neural network built successfully!")
    print("\n📋 Model Architecture:")
    model.summary()
    
    return model

def train_model(model, x_train, y_train_cat, x_test, y_test_cat):
    """Train the neural network"""
    print("\n🚀 Starting model training...")
    
    # Training parameters
    epochs = 10
    batch_size = 32
    validation_split = 0.2
    
    print(f"   Training for {epochs} epochs")
    print(f"   Batch size: {batch_size}")
    print(f"   Validation split: {validation_split}")
    print("\n⏳ Training in progress... (this may take a few minutes)")
    
    # Train the model
    history = model.fit(
        x_train, y_train_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )
    
    print("✅ Training completed!")
    
    return history

def evaluate_model(model, x_test, y_test, y_test_cat):
    """Evaluate the trained model"""
    print("\n📈 Evaluating model performance...")
    
    # Test the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
    
    print(f"✅ Test Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Make predictions
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Classification report
    print("\n📊 Detailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_pred, y_pred_prob

def visualize_training_history(history):
    """Plot training history"""
    print("\n📊 Creating training history visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy During Training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss During Training')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("✅ Saved training history as 'training_history.png'")
    plt.show()

def test_predictions(model, x_test, y_test, num_samples=5):
    """Test the model with sample images"""
    print("\n🔮 Testing predictions on sample images...")
    
    # Select random test images
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    fig.suptitle('Model Predictions on Test Images', fontsize=16)
    
    for i, idx in enumerate(indices):
        # Get image and true label
        img = x_test[idx].reshape(28, 28)
        true_label = y_test[idx]
        
        # Make prediction
        pred_prob = model.predict(x_test[idx:idx+1], verbose=0)
        pred_label = np.argmax(pred_prob)
        confidence = np.max(pred_prob)
        
        # Display image
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}')
        axes[i].axis('off')
        
        # Color code: green for correct, red for incorrect
        if pred_label == true_label:
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ✅\nConf: {confidence:.2f}', 
                             color='green')
        else:
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ❌\nConf: {confidence:.2f}', 
                             color='red')
    
    plt.tight_layout()
    plt.savefig('model_predictions.png', dpi=300, bbox_inches='tight')
    print("✅ Saved predictions as 'model_predictions.png'")
    plt.show()

def save_model(model):
    """Save the trained model"""
    print("\n💾 Saving trained model...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model.save('models/digit_classifier_model.h5')
    print("✅ Model saved as 'models/digit_classifier_model.h5'")
    print("   You can load this model later for predictions!")

def main():
    """Main function to run the complete project"""
    print("🚀 DIGIT CLASSIFIER - NEURAL NETWORK PROJECT (Simplified)")
    print("=" * 60)
    print("This project demonstrates deep learning with neural networks!")
    print("=" * 60)
    
    try:
        # Step 1: Load and prepare data
        x_train, x_test, y_train, y_test, y_train_cat, y_test_cat = load_and_prepare_data()
        
        # Step 2: Explore the dataset
        explore_dataset(x_train, y_train)
        
        # Step 3: Build the neural network
        model = build_neural_network()
        
        # Step 4: Train the model
        history = train_model(model, x_train, y_train_cat, x_test, y_test_cat)
        
        # Step 5: Evaluate the model
        y_pred, y_pred_prob = evaluate_model(model, x_test, y_test, y_test_cat)
        
        # Step 6: Visualize training history
        visualize_training_history(history)
        
        # Step 7: Test predictions
        test_predictions(model, x_test, y_test)
        
        # Step 8: Save the model
        save_model(model)
        
        print("\n🎉 PROJECT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("What you learned:")
        print("✅ How to build a neural network")
        print("✅ How to train deep learning models")
        print("✅ How to process image data")
        print("✅ How to evaluate model performance")
        print("✅ How to make predictions with neural networks")
        print("\n🚀 You now have two working AI projects!")
        print("   Ready to commit to Git and continue learning!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main() 