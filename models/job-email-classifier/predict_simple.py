"""
Simple prediction script for quick email classification
Usage: python predict_simple.py
"""

from predict import EmailClassifier

# Initialize the classifier
classifier = EmailClassifier(model_path='best_bert_email_classifier.pth')

# Your email to classify
email_text = """
Subject: Software Developer Opening

Hi,

We have an opening for a Software Developer position at our company.
Requirements: 3+ years experience in Python, Django, and React.

Please send your resume to jobs@company.com

Best regards,
HR Department
"""

# Make prediction
result = classifier.predict(email_text)

# Print result
classifier.print_prediction(email_text, result)

# Or access the result directly
print(f"\nQuick Result: This email is classified as '{result['prediction']}' with {result['confidence']:.1f}% confidence")
