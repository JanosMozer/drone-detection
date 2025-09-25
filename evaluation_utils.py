import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import tensorflow as tf
from datetime import datetime

# Single model training functions removed - only separate models functions remain

def save_separate_models_results(models, histories, label_encoders, X_train, X_val, X_test, 
                                y_train, y_val, y_test, performance_results, results_dir='results'):
    """Save results for separate models training approach with enhanced logging"""
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Save training configuration and metadata
    training_config = {
        'timestamp': datetime.now().isoformat(),
        'model_count': len(models),
        'task_names': list(models.keys()),
        'data_shapes': {
            'X_train': list(X_train.shape),
            'X_val': list(X_val.shape),
            'X_test': list(X_test.shape)
        },
        'tensorflow_version': tf.__version__,
        'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
        'gpu_devices': [str(gpu) for gpu in tf.config.list_physical_devices('GPU')]
    }
    
    with open(os.path.join(results_dir, 'training_config.json'), 'w') as f:
        json.dump(training_config, f, indent=2)
    
    # Save individual models with detailed info
    model_info = {}
    for task_name, model in models.items():
        model_path = os.path.join(results_dir, f'{task_name}_model.h5')
        model.save(model_path)
        print(f"{task_name} model saved as '{results_dir}/{task_name}_model.h5'")
        
        # Save model architecture summary
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        model_info[task_name] = {
            'model_path': model_path,
            'architecture_summary': model_summary,
            'total_params': int(model.count_params()),
            'trainable_params': int(sum([tf.reduce_prod(var.shape).numpy() for var in model.trainable_variables]))
        }
    
    with open(os.path.join(results_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Save label encoders
    with open(os.path.join(results_dir, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Save training histories with additional metrics
    histories_data = {}
    for task, hist in histories.items():
        hist_data = hist.history.copy()
        # Add training statistics
        hist_data['training_stats'] = {
            'total_epochs': len(hist_data['loss']),
            'best_val_loss_epoch': int(np.argmin(hist_data['val_loss'])) + 1,
            'best_val_acc_epoch': int(np.argmax(hist_data['val_accuracy'])) + 1,
            'final_val_loss': float(hist_data['val_loss'][-1]),
            'final_val_accuracy': float(hist_data['val_accuracy'][-1]),
            'best_val_loss': float(np.min(hist_data['val_loss'])),
            'best_val_accuracy': float(np.max(hist_data['val_accuracy'])),
            'early_stopped': len(hist_data['loss']) < 50  # Assuming max epochs is 50
        }
        histories_data[task] = hist_data
    
    with open(os.path.join(results_dir, 'training_histories.pkl'), 'wb') as f:
        pickle.dump(histories_data, f)
    print(f"Training histories saved as '{results_dir}/training_histories.pkl'")
    
    # Enhanced performance results
    enhanced_performance = performance_results.copy()
    enhanced_performance['training_summary'] = {
        'total_training_time': 'Not tracked',  # Could be added if timing is implemented
        'average_epochs_per_model': np.mean([hist['training_stats']['total_epochs'] 
                                           for hist in histories_data.values()]),
        'models_early_stopped': sum([hist['training_stats']['early_stopped'] 
                                   for hist in histories_data.values()]),
        'overall_best_accuracy': max([results['test_accuracy'] 
                                    for results in performance_results.values()])
    }
    
    with open(os.path.join(results_dir, 'performance_results_separate_models.json'), 'w') as f:
        json.dump(enhanced_performance, f, indent=2)
    
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
    
    # Generate training charts for all models
    generate_separate_models_charts(histories, results_dir)


def generate_separate_models_charts(histories, results_dir):
    """Generate enhanced training charts for separate models"""
    
    plt.figure(figsize=(20, 15))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Distinct colors for each model
    
    for i, (task_name, history) in enumerate(histories.items()):
        color = colors[i % len(colors)]
        
        # Loss plot
        plt.subplot(4, 2, i*2 + 1)
        epochs = range(1, len(history.history['loss']) + 1)
        plt.plot(epochs, history.history['loss'], color=color, linewidth=2, label='Training Loss')
        plt.plot(epochs, history.history['val_loss'], color=color, linewidth=2, linestyle='--', label='Validation Loss')
        
        # Mark best epoch
        best_epoch = np.argmin(history.history['val_loss']) + 1
        best_val_loss = np.min(history.history['val_loss'])
        plt.axvline(x=best_epoch, color='red', linestyle=':', alpha=0.7, label=f'Best Epoch: {best_epoch}')
        plt.scatter(best_epoch, best_val_loss, color='red', s=100, zorder=5)
        
        plt.title(f'{task_name.replace("_", " ").title()} - Loss\n(Stopped at Epoch {len(epochs)})', fontsize=12, fontweight='bold')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy plot
        plt.subplot(4, 2, i*2 + 2)
        plt.plot(epochs, history.history['accuracy'], color=color, linewidth=2, label='Training Accuracy')
        plt.plot(epochs, history.history['val_accuracy'], color=color, linewidth=2, linestyle='--', label='Validation Accuracy')
        
        # Mark best epoch for accuracy
        best_acc_epoch = np.argmax(history.history['val_accuracy']) + 1
        best_val_acc = np.max(history.history['val_accuracy'])
        plt.axvline(x=best_acc_epoch, color='red', linestyle=':', alpha=0.7, label=f'Best Epoch: {best_acc_epoch}')
        plt.scatter(best_acc_epoch, best_val_acc, color='red', s=100, zorder=5)
        
        plt.title(f'{task_name.replace("_", " ").title()} - Accuracy\n(Best Val Acc: {best_val_acc:.4f})', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    chart_path = os.path.join(results_dir, 'training_charts_separate_models.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_separate_models(models, X_test, y_test):
    """Evaluate all separate models on test set"""
    
    all_predictions = {}
    performance_results = {}
    
    for task_name, model in models.items():        
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

    for task_name, results in performance_results.items():
        print(f"{task_name.upper():12} - Accuracy: {results['test_accuracy']:.4f}")
    
    return performance_results, all_predictions
