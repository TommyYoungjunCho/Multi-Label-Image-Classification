# Multi-Label-Image-Classification

**Project Overview**

This project involves the implementation of various deep learning techniques to address a multi-label classification problem. The dataset provided includes images and captions, and the goal is to classify these images into multiple labels. By utilizing a combination of Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTM) networks, and Transformers, the project aims to leverage the strengths of each model to improve classification accuracy. The study explores the integration of natural language processing (NLP) with visual data to enhance model performance.

**Technologies Used**

- **Programming Languages**: Python
- **Libraries**: PyTorch, Transformers, Numpy, Pandas, Matplotlib
- **Tools**: Jupyter Notebook, Google Colaboratory, Kaggle

**Project Achievements**

- **High Classification Accuracy:**
    - Achieved an F1-score of 0.91891 with the hybrid model, demonstrating superior performance in multi-label classification.
- **Innovative Use of Hybrid Models**:
    - Successfully integrated CNNs, LSTMs, and Transformers to create a robust model capable of handling complex multi-label classification tasks.
- **Significant Contributions to Multi-Modal Learning**:
    - Demonstrated the benefits of combining textual and visual data for improving classification accuracy, offering valuable insights for future research in multi-modal learning.

**Period**

- 2024.3 ~ 2024.6

**GitHub Repository**

- https://github.com/TommyYoungjunCho/Multi-Label-Image-Classification

# Project Details

1. **Data Description**:
    - **Dataset**: Custom dataset with images and captions
    - **Images**: Each image has associated labels and a short caption
    - **Classes**: Multi-label classification with various possible labels per image
    - **Evaluation Metric**: Mean F1-Score
    
2. **Data Exploration and Preprocessing**:
    - **Normalization**: Standardized image pixel values to a range of 0 to 1.
    - **Image Transformations**: Applied transformations such as resizing, horizontal and vertical flipping, and random cropping to enhance the dataset.
    
3. **Machine Learning Models**:
    - **Model 1: Convolutional Neural Network (CNN)**:
        - **Architecture**:
            - Multiple convolutional layers followed by pooling layers and fully connected layers.
        - **Activation Functions**: ReLU for hidden layers, Softmax for output layer
        - **Optimization**: Adam optimizer
        - **Loss Function**: Binary Cross-Entropy (BCE) loss for multi-label classification
    - **Model 2: Hybrid Model (CNN + LSTM + Transformer)**:
        - **Architecture**:
            - **CNN Component**: Extracts visual features from images.
            - **LSTM Component**: Processes sequential data from image captions.
            - **Transformer Component**: Captures intricate dependencies and relationships in data.
            - **Architecture**: Combination of CNN layers, LSTM layers, and Transformer encoder layers.
            - **Optimization**: Adam optimizer
            - **Loss Function**: MultiLabelSoftMarginLoss
            
4. **Hyperparameter Tuning**:
    - **Dropout Rates**: Experimented with dropout rates of 0.5 and 0.7 to prevent overfitting.
    - **Batch Sizes**: Tested with batch sizes of 30 and 50 to observe model convergence.
    - **Number of Epochs**: Evaluated performance with 1 and 3 epochs to find the optimal training duration.
    - **Learning Rates**: Compared learning rates of 0.001 and 0.01 to determine the best convergence speed.
    
5. **Result:**
- **Performance Analysis**:
    - Best performance achieved with a combination of 1 epoch, 0.5 dropout rate, batch size of 50, and learning rate of 0.001, yielding an F1-score of 0.91891.
    - Detailed analysis revealed that lower dropout rates and learning rates significantly improved performance.
- **Efficiency Analysis**:
    - Larger batch sizes and lower learning rates contributed to shorter training times.
    - Efficient training processes ensured minimal increase in training time across different epochs.
    
6. **Conclusion:**
- The hybrid model outperformed the standalone CNN model in multi-label classification tasks by leveraging the strengths of each architecture.
- Integrating textual data with visual data improved classification accuracy by providing richer context and information.
- Future work includes refining the hybrid model structure and exploring additional data modalities for further performance enhancement.

## Notion Portfolio Page
- [[Notion Portfolio Page Link](https://magic-taleggio-e52.notion.site/Portfolio-705d90d52e4e451488fb20e3d6653d3b)](#) 
