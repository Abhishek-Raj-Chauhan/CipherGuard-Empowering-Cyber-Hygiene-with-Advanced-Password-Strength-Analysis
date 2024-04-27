# CipherGuard Empowering Cyber Hygiene with Advanced Password Strength Analysis
 Real-time feedback and guidance on password strength using machine learning and deep learning algorithms.

<img width="421" alt="Picture1" src="https://github.com/Abhishek-Raj-Chauhan/CipherGuard-Empowering-Cyber-Hygiene-with-Advanced-Password-Strength-Analysis/assets/100334669/23a3895a-9b5d-4a56-95c9-f819cf22a76b">

# System Architecture							           
The system architecture of our password strength evaluation web application encompasses several interconnected components, each fulfilling a specific role in the process of assessing and enforcing strong password selection. The architecture is designed to seamlessly integrate machine learning & deep learning algorithms, Python libraries, and entropy analyzers to provide real-time analysis of password strength.

•	Data Collection and Preparation: At the outset, the system gathers a diverse dataset comprising passwords of varying strengths—weak, medium, and strong. This dataset serves as the foundation for training and testing our machine learning & deep learning models. The collected dataset undergoes preprocessing techniques to ensure data integrity and uniformity. This involves cleaning, normalization, and feature extraction to facilitate effective model training.

•	Machine Learning & Deep Learning Model Training: The preprocessed dataset is split into training and testing sets using a train-test split technique, with 70% allocated for model training. Various ML and deep learning algorithms, such as decision trees, random forests support vector machines, Ensemble Soft and Hard Voting, Bidirectional LSTM network,  Siamese With positional Encoding and MultiHead are employed to train models on the labeled dataset. Each algorithm is evaluated to determine its effectiveness in predicting password strength accurately.

•	Model Testing and Evaluation: The remaining 30% of the dataset is utilized for testing and evaluating the performance of the trained models. This testing phase assesses the model's ability to generalize to unseen data and accurately classify password strength. Metrics such as accuracy, precision, recall, and F1 score are calculated to gauge the efficacy of each model in accurately predicting password strength.

•	Real-time Analysis and Enforcement: Once the model is trained and validated, it is integrated into the web application's backend to enable real-time password strength analysis. When users attempt to create or update their passwords, the system utilizes the trained model to evaluate the strength of the input password. If the password is deemed weak or insufficiently strong, users are prompted to choose a stronger password, thereby enforcing robust security practices.

•	User Interface and Frontend: The frontend of the web application provides an intuitive user interface for interacting with the system. Users are guided through the process of creating or updating their passwords, with real-time feedback on the strength of their chosen passwords. Visual cues and prompts are provided to assist users in selecting strong passwords and complying with security guidelines.

•	Communication and Data Flow: Communication between the frontend and backend is facilitated through APIs or HTTP requests, ensuring seamless interaction and data exchange. User input, including passwords, is securely transmitted to the backend for analysis, while feedback on password strength is relayed back to the frontend for display to the user.

•	Scalability and Performance: The architecture is designed to be scalable, allowing for the addition of new features, algorithms, and datasets as the application evolves. Emphasis is placed on optimizing performance to ensure real-time analysis and response, even under high user loads.

