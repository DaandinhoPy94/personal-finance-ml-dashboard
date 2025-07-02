"""
Machine Learning models voor Personal Finance Dashboard
Automatische categorisatie en anomaly detection
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from textblob import TextBlob
import re

class TransactionCategorizer:
    """
    AI model voor automatische transactie categorisatie
    Leert van beschrijvingen om categorieën te voorspellen
    """
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.categories = []
        
    def clean_description(self, text):
        """
        Maakt transactie beschrijving schoon voor ML
        
        Args:
            text (str): Ruwe beschrijving
            
        Returns:
            str: Schone tekst voor ML
        """
        if pd.isna(text) or text == "":
            return ""
            
        # Converteer naar lowercase
        text = str(text).lower()
        
        # Verwijder speciale karakters maar behoud spaties
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Verwijder dubbele spaties
        text = re.sub(r'\s+', ' ', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def prepare_training_data(self, df, min_samples_per_category=2):
        """
        Bereidt data voor voor ML training
        
        Args:
            df (DataFrame): Transactie data met 'description' en 'category'
            min_samples_per_category (int): Minimum aantal samples per categorie
            
        Returns:
            tuple: (X, y) voor training
        """
        # Check of required kolommen bestaan
        if 'description' not in df.columns or 'category' not in df.columns:
            raise ValueError("DataFrame moet 'description' en 'category' kolommen hebben")
        
        # Filter out missing values
        df_clean = df.dropna(subset=['description', 'category'])
        
        # Clean descriptions
        df_clean['description_clean'] = df_clean['description'].apply(self.clean_description)
        
        # Remove empty descriptions
        df_clean = df_clean[df_clean['description_clean'] != ""]
        
        # Count samples per category
        category_counts = df_clean['category'].value_counts()
        
        # Filter out categories with too few samples
        valid_categories = category_counts[category_counts >= min_samples_per_category].index
        
        if len(valid_categories) == 0:
            raise ValueError(f"Geen categorieën hebben minimaal {min_samples_per_category} samples!")
        
        # Filter data to only include valid categories
        df_filtered = df_clean[df_clean['category'].isin(valid_categories)]
        
        # Report filtering results
        total_categories = len(category_counts)
        filtered_categories = len(valid_categories)
        removed_categories = category_counts[category_counts < min_samples_per_category]
        
        print(f"📊 Categorie filtering:")
        print(f"   Totaal categorieën: {total_categories}")
        print(f"   Bruikbare categorieën: {filtered_categories}")
        if len(removed_categories) > 0:
            print(f"   Weggelaten (te weinig data): {list(removed_categories.index)}")
        
        # Store only valid categories
        self.categories = sorted(df_filtered['category'].unique())
        
        X = df_filtered['description_clean'].values
        y = df_filtered['category'].values
        
        return X, y
    
    def train(self, df, test_size=0.2, random_state=42):
        """
        Traint het categorisatie model
        
        Args:
            df (DataFrame): Training data
            test_size (float): Percentage voor testing
            random_state (int): Voor reproduceerbare resultaten
            
        Returns:
            dict: Training resultaten en metrics
        """
        print("🤖 Starting model training...")
        
        # Prepare data
        X, y = self.prepare_training_data(df, min_samples_per_category=2)
        
        if len(X) < 10:
            raise ValueError("Niet genoeg data voor training (minimum 10 samples)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Create ML pipeline
        # TfidfVectorizer: Converteerd tekst naar numbers
        # MultinomialNB: Naive Bayes classifier (goed voor tekst)
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1000,  # Maximum 1000 woorden
                stop_words='english',  # Negeer veel voorkomende woorden
                ngram_range=(1, 2)  # Gebruik 1 en 2 woord combinaties
            )),
            ('classifier', MultinomialNB(alpha=1.0))
        ])
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Test accuracy
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate detailed report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.is_trained = True
        
        training_results = {
            'accuracy': accuracy,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'categories': self.categories,
            'classification_report': report
        }
        
        print(f"✅ Model trained! Accuracy: {accuracy:.2%}")
        
        return training_results
    
    def predict_category(self, description):
        """
        Voorspelt categorie voor een nieuwe beschrijving
        
        Args:
            description (str): Transactie beschrijving
            
        Returns:
            tuple: (predicted_category, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model is nog niet getraind! Roep eerst .train() aan.")
        
        # Clean description
        description_clean = self.clean_description(description)
        
        if description_clean == "":
            return "Unknown", 0.0
        
        # Predict
        predicted_category = self.model.predict([description_clean])[0]
        
        # Get confidence (probability)
        probabilities = self.model.predict_proba([description_clean])[0]
        confidence = max(probabilities)
        
        return predicted_category, confidence
    
    def predict_batch(self, descriptions):
        """
        Voorspelt categorieën voor meerdere beschrijvingen tegelijk
        
        Args:
            descriptions (list): Lijst van beschrijvingen
            
        Returns:
            DataFrame: Beschrijvingen met voorspelde categorieën en confidence
        """
        if not self.is_trained:
            raise ValueError("Model is nog niet getraind!")
        
        results = []
        
        for desc in descriptions:
            category, confidence = self.predict_category(desc)
            results.append({
                'description': desc,
                'predicted_category': category,
                'confidence': confidence
            })
        
        return pd.DataFrame(results)
    
    def save_model(self, filepath='models/transaction_categorizer.joblib'):
        """
        Slaat getraind model op
        """
        if not self.is_trained:
            raise ValueError("Geen model om op te slaan!")
        
        # Maak models directory
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model en metadata
        model_data = {
            'pipeline': self.model,
            'categories': self.categories,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model opgeslagen: {filepath}")
    
    def load_model(self, filepath='models/transaction_categorizer.joblib'):
        """
        Laadt een opgeslagen model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file niet gevonden: {filepath}")
        
        # Load model data
        model_data = joblib.load(filepath)
        
        self.model = model_data['pipeline']
        self.categories = model_data['categories']
        self.is_trained = model_data['is_trained']
        
        print(f"📁 Model geladen: {filepath}")

class AnomalyDetector:
    """
    Detecteert ongewone uitgaven patronen
    """
    
    def __init__(self):
        self.category_stats = {}
        
    def analyze_spending_patterns(self, df):
        """
        Analyseert uitgaven patronen per categorie
        """
        # Filter alleen uitgaven
        spending_df = df[df['amount'] < 0].copy()
        spending_df['amount_abs'] = spending_df['amount'].abs()
        
        # Bereken statistieken per categorie
        stats = {}
        
        for category in spending_df['category'].unique():
            cat_data = spending_df[spending_df['category'] == category]['amount_abs']
            
            stats[category] = {
                'mean': cat_data.mean(),
                'std': cat_data.std(),
                'median': cat_data.median(),
                'q75': cat_data.quantile(0.75),
                'q95': cat_data.quantile(0.95),
                'max': cat_data.max(),
                'count': len(cat_data)
            }
        
        self.category_stats = stats
        return stats
    
    def detect_anomalies(self, df, threshold_multiplier=2.5):
        """
        Detecteert ongewone uitgaven
        
        Args:
            df: DataFrame met transacties
            threshold_multiplier: Hoeveel standaarddeviaties = anomalie
            
        Returns:
            DataFrame: Transacties die anomalieën zijn
        """
        if not self.category_stats:
            self.analyze_spending_patterns(df)
        
        anomalies = []
        
        for _, transaction in df[df['amount'] < 0].iterrows():
            category = transaction['category']
            amount = abs(transaction['amount'])
            
            if category in self.category_stats:
                stats = self.category_stats[category]
                
                # Z-score anomaly detection
                if stats['std'] > 0:  # Avoid division by zero
                    z_score = abs((amount - stats['mean']) / stats['std'])
                    
                    if z_score > threshold_multiplier:
                        anomaly_info = transaction.to_dict()
                        anomaly_info['z_score'] = z_score
                        anomaly_info['category_mean'] = stats['mean']
                        anomaly_info['amount_abs'] = amount
                        anomalies.append(anomaly_info)
        
        return pd.DataFrame(anomalies)