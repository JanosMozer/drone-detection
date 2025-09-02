import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import tensorflow as tf

def save_training_results(model, history, label_encoders, X_train, X_val, X_test, 
                         y_train, y_val, y_test, results_dir='results'):
    """Save all training results, processed data, and generate charts"""
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the model and encoders
    model.save(os.path.join(results_dir, 'multi_output_aircraft_model.h5'))
    print(f"\nModel saved as '{results_dir}/multi_output_aircraft_model.h5'")
    
    with open(os.path.join(results_dir, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"Label encoders saved as '{results_dir}/label_encoders.pkl'")
    
    # Save training history
    with open(os.path.join(results_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Training history saved as '{results_dir}/training_history.pkl'")
    
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
    
    # Generate and save training charts
    generate_training_charts(history, results_dir)
    
    # Save detailed results summary
    save_results_summary(model, history, results_dir)
    
    print(f"\nAll results saved in '{results_dir}/' directory!")

def generate_training_charts(history, results_dir):
    """Generate and save training charts"""
    
    plt.figure(figsize=(15, 10))
    
    # Plot training history for each task
    tasks = ['is_aircraft', 'engtype', 'engnum', 'fueltype']
    metrics = ['loss', 'accuracy']
    
    for i, task in enumerate(tasks):
        for j, metric in enumerate(metrics):
            plt.subplot(4, 2, i*2 + j + 1)
            
            if metric == 'loss':
                plt.plot(history.history[f'{task}_{metric}'], label=f'Training {metric}')
                plt.plot(history.history[f'val_{task}_{metric}'], label=f'Validation {metric}')
                plt.title(f'{task.replace("_", " ").title()} - {metric.title()}')
                plt.ylabel(metric.title())
            else:
                plt.plot(history.history[f'{task}_{metric}'], label=f'Training {metric}')
                plt.plot(history.history[f'val_{task}_{metric}'], label=f'Validation {metric}')
                plt.title(f'{task.replace("_", " ").title()} - {metric.title()}')
                plt.ylabel(metric.title())
            
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_charts.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training charts saved as '{results_dir}/training_charts.png'")

def save_results_summary(model, history, results_dir):
    """Save detailed results summary"""
    
    # Get model summary as string
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    model_summary = '\n'.join(summary_list)
    
    results_summary = {
        'total_parameters': model.count_params(),
        'training_epochs': len(history.history['loss']),
        'final_training_loss': history.history['loss'][-1],
        'final_validation_loss': history.history['val_loss'][-1],
        'model_architecture': model_summary
    }
    
    with open(os.path.join(results_dir, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    print(f"Results summary saved as '{results_dir}/results_summary.json'")

def evaluate_model_performance(model, X_test, y_test, results_dir='results'):
    """Evaluate model performance and save results"""
    
    # Get predictions
    predictions = model.predict(X_test, verbose=0)
    pred_is_aircraft = (predictions[0] > 0.5).astype(int).flatten()
    pred_engtype = np.argmax(predictions[1], axis=1)
    pred_engnum = np.argmax(predictions[2], axis=1)
    pred_fueltype = np.argmax(predictions[3], axis=1)
    
    # Evaluate each task
    print("\n=== RESULTS ===")
    
    # is_aircraft task
    print("Aircraft Detection:")
    aircraft_mask_test = y_test['is_aircraft'] == 1
    acc_aircraft = (pred_is_aircraft == y_test['is_aircraft']).mean()
    print(f"  Accuracy: {acc_aircraft:.4f}")
    
    # Initialize accuracy variables
    acc_engtype = None
    acc_engnum = None
    acc_fueltype = None
    
    # engtype task (only for aircraft)
    if aircraft_mask_test.sum() > 0:
        engtype_true = y_test['engtype'][aircraft_mask_test]
        engtype_pred = pred_engtype[aircraft_mask_test]
        acc_engtype = (engtype_true == engtype_pred).mean()
        print(f"Engine Type Accuracy: {acc_engtype:.4f}")
        
        # engnum task (only for aircraft)
        engnum_true = y_test['engnum'][aircraft_mask_test]
        engnum_pred = pred_engnum[aircraft_mask_test]
        acc_engnum = (engnum_true == engnum_pred).mean()
        print(f"Engine Number Accuracy: {acc_engnum:.4f}")
        
        # fueltype task (only for aircraft)
        fueltype_true = y_test['fueltype'][aircraft_mask_test]
        fueltype_pred = pred_fueltype[aircraft_mask_test]
        acc_fueltype = (fueltype_true == fueltype_pred).mean()
        print(f"Fuel Type Accuracy: {acc_fueltype:.4f}")
    
    # Save performance results
    performance_results = {
        'test_accuracy_aircraft': float(acc_aircraft),
        'test_accuracy_engtype': float(acc_engtype) if acc_engtype is not None else None,
        'test_accuracy_engnum': float(acc_engnum) if acc_engnum is not None else None,
        'test_accuracy_fueltype': float(acc_fueltype) if acc_fueltype is not None else None
    }
    
    with open(os.path.join(results_dir, 'performance_results.json'), 'w') as f:
        json.dump(performance_results, f, indent=2)
    print(f"Performance results saved as '{results_dir}/performance_results.json'")
    
    return performance_results
