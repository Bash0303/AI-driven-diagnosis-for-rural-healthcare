import pandas as pd
import numpy as np
import joblib
import re
import os
import sys
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    """Train the disease diagnosis model and return output as string"""
    output_capture = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_capture
    
    try:
        print("=" * 60)
        print("AI DISEASE DIAGNOSIS MODEL TRAINING")
        print("=" * 60)
        
        # Load dataset
        print("\n[1] Loading dataset 'Symptom2disease.csv'...")
        try:
            # Try to load the dataset
            if not os.path.exists('Symptom2disease.csv'):
                # Try alternative path
                if os.path.exists('./Symptom2disease.csv'):
                    df = pd.read_csv('./Symptom2disease.csv')
                else:
                    print("[ERROR] File 'Symptom2disease.csv' not found!")
                    print("[INFO] Make sure the CSV file is in the same folder")
                    return output_capture.getvalue()
            else:
                df = pd.read_csv('Symptom2disease.csv')
            
            # Show dataset info
            print(f"Dataset shape: {df.shape}")
            print(f"Columns found: {list(df.columns)}")
            
            # Handle different column names
            if 'text' in df.columns and 'label' in df.columns:
                df = df[['text', 'label']].copy()
            elif 'symptom' in df.columns and 'disease' in df.columns:
                df = df.rename(columns={'symptom': 'text', 'disease': 'label'})
            elif len(df.columns) >= 2:
                # Use first two columns
                df = df.iloc[:, :2]
                df.columns = ['text', 'label']
                print("[INFO] Using first two columns as text and label")
            
            # Clean data
            df = df.dropna()
            df['text'] = df['text'].astype(str).str.strip()
            df['label'] = df['label'].astype(str).str.strip()
            
            print(f"[SUCCESS] Loaded {len(df)} records")
            print(f"[SUCCESS] Found {df['label'].nunique()} unique diseases")
            
            # Show sample diseases
            sample_diseases = list(df['label'].unique())[:10]
            print(f"[INFO] Sample diseases: {', '.join(sample_diseases)}")
            
        except FileNotFoundError:
            print("[ERROR] File 'Symptom2Disease.csv' not found!")
            print("[INFO] Make sure the CSV file is in the same folder")
            return output_capture.getvalue()
        except Exception as e:
            print(f"[ERROR] Could not load data: {e}")
            return output_capture.getvalue()
        
        # Preprocessing function
        def preprocess_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-z\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        # Preprocess text
        print("\n[2] Preprocessing text data...")
        df['cleaned_text'] = df['text'].apply(preprocess_text)
        
        # Encode labels
        print("[3] Encoding disease labels...")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['label'])
        
        # Vectorize text
        print("[4] Vectorizing text with TF-IDF...")
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(df['cleaned_text'])
        
        # Split data
        print("[5] Splitting data (80% train, 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"[INFO] Training samples: {X_train.shape[0]}")
        print(f"[INFO] Testing samples: {X_test.shape[0]}")
        
        # Train model
        print("\n[6] Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"[SUCCESS] Model trained successfully!")
        print(f"[RESULT] Accuracy: {accuracy:.2%}")
        
        # Classification report
        print("\n[7] Top 10 Diseases Performance:")
        all_labels = label_encoder.classes_
        
        # Get classification report for all classes
        report = classification_report(y_test, y_pred, 
                                      target_names=all_labels,
                                      output_dict=True,
                                      zero_division=0)
        
        # Show top 10 diseases by F1-score
        f1_scores = []
        for disease in all_labels:
            if disease in report:
                f1_scores.append((disease, report[disease]['f1-score']))
        
        # Sort by F1-score
        f1_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("Disease           | Precision | Recall   | F1-Score")
        print("-" * 50)
        for disease, f1 in f1_scores[:10]:  # Show top 10
            precision = report[disease]['precision']
            recall = report[disease]['recall']
            print(f"{disease[:15]:15} | {precision:.3f}     | {recall:.3f}    | {f1:.3f}")
        
        # Save model
        print("\n[8] Saving model...")
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
            print("[INFO] Created 'models' directory")
        
        model_data = {
            'model': model,
            'vectorizer': vectorizer,
            'label_encoder': label_encoder
        }
        
        joblib.dump(model_data, 'models/trained_model.pkl')
        print(f"[SAVE] Model saved to: models/trained_model.pkl")
        
        # Save disease list
        diseases = label_encoder.classes_
        disease_df = pd.DataFrame(diseases, columns=['disease'])
        disease_df.to_csv('models/diseases.csv', index=False)
        print(f"[SAVE] Disease list saved: models/diseases.csv")
        print(f"[INFO] Model can detect {len(diseases)} diseases")
        
        # Test with samples
        print("\n[9] Testing with sample symptoms:")
        test_samples = [
            "I have fever and headache with body pain",
            "Skin rash with itching and red patches on arms",
            "Cough with chest pain and difficulty breathing",
            "Stomach pain with vomiting and diarrhea"
        ]
        
        for i, text in enumerate(test_samples, 1):
            cleaned = preprocess_text(text)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)
            disease = label_encoder.inverse_transform(prediction)[0]
            proba = model.predict_proba(vectorized)[0]
            confidence = max(proba) * 100
            
            print(f"\n  Sample {i}: '{text[:50]}...'")
            print(f"  → Predicted: {disease}")
            print(f"  → Confidence: {confidence:.1f}%")
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Go to 'AI Diagnosis' page in the app")
        print("2. Enter symptoms to get predictions")
        print("3. Model is ready for use!")
        
        return output_capture.getvalue()
        
    except Exception as e:
        print(f"[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return output_capture.getvalue()
    finally:
        sys.stdout = original_stdout

if __name__ == "__main__":
    output = train_model()
    print(output)