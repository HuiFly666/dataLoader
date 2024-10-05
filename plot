def plot_metrics(metrics_history, train_times, test_times):
    epochs = range(1, len(metrics_history['accuracy']) + 1)
    
    # 绘制准确率
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, metrics_history['accuracy'], label='Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # 绘制其他指标
    plt.subplot(2, 1, 2)
    plt.plot(epochs, metrics_history['precision'], label='Precision')
    plt.plot(epochs, metrics_history['recall'], label='Recall')
    plt.plot(epochs, metrics_history['f1'], label='F1 Score')
    plt.plot(epochs, metrics_history['roc_auc'], label='ROC AUC')
    plt.plot(epochs, metrics_history['balanced_accuracy'], label='Balanced Accuracy')
    plt.title('Metrics Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig('model_metrics.png')
    plt.show()

    # 绘制训练时间和测试时间
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_times, label='Train Time')
    plt.plot(epochs, test_times, label='Test Time')
    plt.title('Train and Test Time Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Time (seconds)')
    plt.legend()

    # 保存时间图像
    plt.tight_layout()
    plt.savefig('train_test_time.png')
    plt.show()

# 在训练完成后调用绘制函数
plot_metrics(metrics_history, train_times, test_times)
