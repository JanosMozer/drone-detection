import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import tensorflow as tf

# Single model training functions removed - only separate models functions remain

def save_separate_models_results(models, histories, label_encoders, X_train, X_val, X_test, 
                                y_train, y_val, y_test, performance_results, results_dir='results'):
    """Save results for separate models training approach"""
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Save individual models
    for task_name, model in models.items():
        model.save(os.path.join(results_dir, f'{task_name}_model.h5'))
        print(f"{task_name} model saved as '{results_dir}/{task_name}_model.h5'")
    
    # Save label encoders
    with open(os.path.join(results_dir, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"Label encoders saved as '{results_dir}/label_encoders.pkl'")
    
    # Save training histories
    with open(os.path.join(results_dir, 'training_histories.pkl'), 'wb') as f:
        pickle.dump({task: hist.history for task, hist in histories.items()}, f)
    print(f"Training histories saved as '{results_dir}/training_histories.pkl'")
    
    # Save performance results
    with open(os.path.join(results_dir, 'performance_results_separate_models.json'), 'w') as f:
        json.dump(performance_results, f, indent=2)
    print(f"Performance results saved as '{results_dir}/performance_results_separate_models.json'")
    
    # Save processed data
    np.save(os.path.join(results_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(results_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(results_dir, 'X_test.npy'), X_test)
    
    with open(os.path.join(results_dir, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y_train, f)
    with open(os.path.join(results_dir, 'y_val.pkl'), 'wb') as f:
        pickle.dump(y_val, f)
    with open(os.path.join(results_dir, 'y_test.pkl'), 'wb') as f:
        pickle.dump(y_test, f)
    print(f"Processed data saved in '{results_dir}/'")
    
    # Generate training charts for all models
    generate_separate_models_charts(histories, results_dir)
    
    print(f"\nAll results saved in '{results_dir}/' directory!")

def generate_separate_models_charts(histories, results_dir):
    """Generate training charts for separate models"""
    
    plt.figure(figsize=(20, 15))
    for i, (task_name, history) in enumerate(histories.items()):
        # Loss plot
        plt.subplot(4, 2, i*2 + 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{task_name.replace("_", " ").title()} - Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy plot
        plt.subplot(4, 2, i*2 + 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{task_name.replace("_", " ").title()} - Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_charts_separate_models.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training charts saved as '{results_dir}/training_charts_separate_models.png'")

def evaluate_separate_models(models, X_test, y_test):
    """Evaluate all separate models on test set"""
    
    print(f"\n{'='*60}")
    print("EVALUATING ALL MODELS ON TEST SET")
    print(f"{'='*60}")
    
    all_predictions = {}
    performance_results = {}
    
    for task_name, model in models.items():
        print(f"\nEvaluating {task_name.upper()} model...")
        
        # Prepare test data for this task
        if task_name == 'is_aircraft':
            X_task_test = X_test
            y_task_test = y_test[task_name]
        elif task_name == 'engnum':
            # For engnum, exclude samples with -1 (4-engine aircraft)
            valid_mask_test = y_test[task_name] != -1
            X_task_test = X_test[valid_mask_test]
            y_task_test = y_test[task_name][valid_mask_test]
        else:
            # For engtype and fueltype, use aircraft samples
            valid_mask_test = y_test['is_aircraft'] == 1
            X_task_test = X_test[valid_mask_test]
            y_task_test = y_test[task_name][valid_mask_test]
        
        # Evaluate model
        test_loss, test_acc = model.evaluate(X_task_test, y_task_test, verbose=0)
        print(f"{task_name.upper()} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
        # Get predictions
        predictions = model.predict(X_task_test, verbose=0)
        
        if task_name == 'engtype':
            pred_labels = np.argmax(predictions, axis=1)
        else:
            pred_labels = (predictions > 0.5).astype(int).flatten()
        
        all_predictions[task_name] = pred_labels
        performance_results[task_name] = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc)
        }
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    for task_name, results in performance_results.items():
        print(f"{task_name.upper():12} - Accuracy: {results['test_accuracy']:.4f}")
    
    return performance_results, all_predictions
