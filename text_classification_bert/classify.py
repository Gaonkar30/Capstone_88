import torch

import pandas as pd

import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.utils.class_weight import compute_class_weight

from transformers import (

    DistilBertTokenizer,

    DistilBertForSequenceClassification,

    Trainer,

    TrainingArguments,

    EarlyStoppingCallback

)

from torch.utils.data import Dataset, WeightedRandomSampler

import warnings

import random

import re

warnings.filterwarnings('ignore')



# ============================================

# DATA AUGMENTATION TECHNIQUES

# ============================================



class TextAugmenter:

    """Augment text data to balance classes"""

    

    @staticmethod

    def synonym_replacement(text, n=2):

        """Simple synonym replacement (basic version)"""

        words = text.split()

        if len(words) < 3:

            return text

        

        # Simple word shuffling for augmentation

        for _ in range(n):

            idx = random.randint(0, len(words) - 1)

            # Swap with adjacent word

            if idx < len(words) - 1:

                words[idx], words[idx + 1] = words[idx + 1], words[idx]

        

        return ' '.join(words)

    

    @staticmethod

    def random_deletion(text, p=0.1):

        """Randomly delete words with probability p"""

        words = text.split()

        if len(words) == 1:

            return text

        

        new_words = [word for word in words if random.random() > p]

        

        if len(new_words) == 0:

            return random.choice(words)

        

        return ' '.join(new_words)

    

    @staticmethod

    def random_swap(text, n=2):

        """Randomly swap n pairs of words"""

        words = text.split()

        if len(words) < 2:

            return text

        

        for _ in range(n):

            idx1, idx2 = random.sample(range(len(words)), 2)

            words[idx1], words[idx2] = words[idx2], words[idx1]

        

        return ' '.join(words)

    

    @staticmethod

    def back_translation_simulation(text):

        """Simulate back-translation by slight paraphrasing"""

        # This is a simplified version - real back-translation uses translation APIs

        words = text.split()

        if len(words) < 3:

            return text

        

        # Randomly shuffle chunks

        chunk_size = max(2, len(words) // 3)

        chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]

        random.shuffle(chunks)

        

        return ' '.join([' '.join(chunk) for chunk in chunks])

    

    @classmethod

    def augment_text(cls, text, num_augmentations=1):

        """Generate augmented versions of text"""

        augmented = [text]  # Include original

        

        methods = [

            cls.synonym_replacement,

            cls.random_deletion,

            cls.random_swap,

            cls.back_translation_simulation

        ]

        

        for _ in range(num_augmentations):

            method = random.choice(methods)

            aug_text = method(text)

            augmented.append(aug_text)

        

        return augmented



# ============================================

# DATA LOADING WITH AUGMENTATION

# ============================================



def load_data_from_folders(root_folder, min_samples=10, augment=True):

    """

    Load data from folder structure with automatic augmentation for small classes

    

    Args:

        root_folder: Path to root folder containing category subfolders

        min_samples: Minimum samples per category after augmentation

        augment: Whether to augment underrepresented classes

    """

    root_path = Path(root_folder)

    

    texts = []

    labels = []

    

    category_folders = [f for f in root_path.iterdir() if f.is_dir()]

    

    if len(category_folders) == 0:

        raise ValueError(f"No subdirectories found in {root_folder}")

    

    print(f"Found {len(category_folders)} categories:")

    for folder in category_folders:

        print(f"  - {folder.name}")

    

    label_mapping = {folder.name: idx for idx, folder in enumerate(sorted(category_folders))}

    

    # First pass: collect all data

    category_data = {cat: [] for cat in label_mapping.keys()}

    

    for category_folder in category_folders:

        category_name = category_folder.name

        text_files = list(category_folder.glob('*.txt'))

        

        print(f"\nReading {len(text_files)} files from '{category_name}'...")

        

        for text_file in text_files:

            try:

                with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:

                    content = f.read().strip()

                    if content:

                        category_data[category_name].append(content)

            except Exception as e:

                print(f"  Warning: Could not read {text_file.name}: {e}")

    

    # Display original distribution

    print("\n" + "="*60)

    print("ORIGINAL CATEGORY DISTRIBUTION:")

    print("="*60)

    for cat, data in sorted(category_data.items()):

        print(f"{cat:30s}: {len(data):4d} samples")

    

    # Augment underrepresented classes

    if augment:

        print("\n" + "="*60)

        print("AUGMENTING UNDERREPRESENTED CLASSES:")

        print("="*60)

        

        augmenter = TextAugmenter()

        

        for category_name, category_texts in category_data.items():

            original_count = len(category_texts)

            

            if original_count < min_samples and original_count > 0:

                needed = min_samples - original_count

                augmentations_per_sample = (needed // original_count) + 1

                

                print(f"\n{category_name}:")

                print(f"  Original: {original_count} samples")

                print(f"  Target: {min_samples} samples")

                print(f"  Generating {augmentations_per_sample} augmentations per sample...")

                

                augmented_texts = []

                for text in category_texts:

                    aug_versions = augmenter.augment_text(text, augmentations_per_sample)

                    augmented_texts.extend(aug_versions[1:])  # Skip original

                

                # Add augmented texts (limit to needed amount)

                category_data[category_name].extend(augmented_texts[:needed])

                print(f"  After augmentation: {len(category_data[category_name])} samples")

    

    # Filter out categories that still have too few samples

    min_required = 3

    valid_categories = {cat: texts for cat, texts in category_data.items() if len(texts) >= min_required}

    

    if len(valid_categories) < len(category_data):

        removed = set(category_data.keys()) - set(valid_categories.keys())

        print(f"\n⚠️ Removing {len(removed)} categories with < {min_required} samples after augmentation:")

        for cat in removed:

            print(f"  - {cat}: {len(category_data[cat])} samples")

    

    # Rebuild label mapping with only valid categories

    label_mapping = {cat: idx for idx, cat in enumerate(sorted(valid_categories.keys()))}

    

    # Build final dataset

    print("\n" + "="*60)

    print("FINAL CATEGORY DISTRIBUTION:")

    print("="*60)

    

    for category_name in sorted(valid_categories.keys()):

        category_texts = valid_categories[category_name]

        category_label = label_mapping[category_name]

        print(f"{category_name:30s}: {len(category_texts):4d} samples")

        

        for text in category_texts:

            texts.append(text)

            labels.append(category_label)

    

    df = pd.DataFrame({

        'text': texts,

        'category': [list(label_mapping.keys())[list(label_mapping.values()).index(l)] for l in labels],

        'label_encoded': labels

    })

    

    print(f"\nTotal samples: {len(df)}")

    print(f"Number of classes: {len(label_mapping)}")

    

    # Verify labels are in correct range

    assert df['label_encoded'].min() == 0, "Labels should start from 0"

    assert df['label_encoded'].max() == len(label_mapping) - 1, f"Max label should be {len(label_mapping)-1}"

    

    return df, label_mapping



# ============================================

# CUSTOM DATASET WITH CLASS WEIGHTS

# ============================================



class TextDataset(Dataset):

    """Custom dataset for text classification"""

    def __init__(self, texts, labels, tokenizer, max_len=512):

        self.texts = texts

        self.labels = labels

        self.tokenizer = tokenizer

        self.max_len = max_len

    

    def __len__(self):

        return len(self.texts)

    

    def __getitem__(self, idx):

        text = str(self.texts[idx])

        label = self.labels[idx]

        

        encoding = self.tokenizer.encode_plus(

            text,

            add_special_tokens=True,

            max_length=self.max_len,

            padding='max_length',

            truncation=True,

            return_attention_mask=True,

            return_tensors='pt'

        )

        

        return {

            'input_ids': encoding['input_ids'].flatten(),

            'attention_mask': encoding['attention_mask'].flatten(),

            'labels': torch.tensor(label, dtype=torch.long)

        }



# ============================================

# IMPROVED CLASSIFIER WITH CLASS BALANCING

# ============================================



class ImprovedTextClassifier:

    """Enhanced text classifier with class balancing"""

    

    def __init__(self, num_labels, model_name='distilbert-base-uncased'):

        self.model_name = model_name

        self.num_labels = num_labels

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        

        print(f"Using device: {self.device}")

        

        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

        self.model = DistilBertForSequenceClassification.from_pretrained(

            model_name,

            num_labels=num_labels

        )

        self.model.to(self.device)

        self.class_weights = None

    

    def prepare_datasets(self, df, label_mapping=None, test_size=0.2, val_size=0.1):

        """Split data and create PyTorch datasets with class weights"""

        

        texts = df['text'].values

        labels = df['label_encoded'].values

        

        # Verify labels are valid

        unique_labels = np.unique(labels)

        print(f"\nLabel validation:")

        print(f"  Unique labels: {unique_labels}")

        print(f"  Expected range: 0 to {self.num_labels - 1}")

        print(f"  Min label: {labels.min()}")

        print(f"  Max label: {labels.max()}")

        

        assert labels.min() >= 0, "Labels must be >= 0"

        assert labels.max() < self.num_labels, f"Labels must be < {self.num_labels}, found {labels.max()}"

        

        # Calculate class weights for imbalanced data

        class_weights = compute_class_weight(

            'balanced',

            classes=np.unique(labels),

            y=labels

        )

        self.class_weights = torch.FloatTensor(class_weights).to(self.device)

        

        print("\nClass weights (higher = more importance to rare classes):")

        for i, weight in enumerate(class_weights):

            cat_name = list(label_mapping.keys())[list(label_mapping.values()).index(i)]

            count = np.sum(labels == i)

            print(f"  {cat_name:30s}: {weight:.3f} ({count} samples)")

        

        # Split data with stratification

        try:

            X_train, X_temp, y_train, y_temp = train_test_split(

                texts, labels, test_size=(test_size + val_size), random_state=42, stratify=labels

            )

            

            X_val, X_test, y_val, y_test = train_test_split(

                X_temp, y_temp, test_size=(test_size/(test_size + val_size)), random_state=42, stratify=y_temp

            )

        except ValueError as e:

            print(f"\n⚠️ Warning: Stratification failed, using random split: {e}")

            X_train, X_temp, y_train, y_temp = train_test_split(

                texts, labels, test_size=(test_size + val_size), random_state=42

            )

            

            X_val, X_test, y_val, y_test = train_test_split(

                X_temp, y_temp, test_size=(test_size/(test_size + val_size)), random_state=42

            )

        

        print(f"\nDataset split:")

        print(f"  Train samples: {len(X_train)}")

        print(f"  Validation samples: {len(X_val)}")

        print(f"  Test samples: {len(X_test)}")

        

        train_dataset = TextDataset(X_train, y_train, self.tokenizer)

        val_dataset = TextDataset(X_val, y_val, self.tokenizer)

        test_dataset = TextDataset(X_test, y_test, self.tokenizer)

        

        return train_dataset, val_dataset, test_dataset

    

    def train(self, train_dataset, val_dataset, output_dir='./results', epochs=5):

        """Fine-tune with class weights"""

        

        # Custom loss function with class weights

        class WeightedTrainer(Trainer):

            def __init__(self, class_weights, *args, **kwargs):

                super().__init__(*args, **kwargs)

                self.class_weights = class_weights

            

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

                labels = inputs.pop("labels")

                outputs = model(**inputs)

                logits = outputs.logits

                

                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)

                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

                

                return (loss, outputs) if return_outputs else loss

        

        training_args = TrainingArguments(

            output_dir=output_dir,

            num_train_epochs=epochs,

            per_device_train_batch_size=8,  # Reduced for better learning

            per_device_eval_batch_size=16,

            learning_rate=2e-5,  # Standard learning rate

            warmup_steps=500,

            weight_decay=0.01,

            logging_dir='./logs',

            logging_steps=50,

            eval_strategy='epoch',

            save_strategy='epoch',

            load_best_model_at_end=True,

            metric_for_best_model='eval_loss',

            greater_is_better=False,

            save_total_limit=2,

            report_to='none'

        )

        

        def compute_metrics(pred):

            labels = pred.label_ids

            preds = pred.predictions.argmax(-1)

            acc = accuracy_score(labels, preds)

            return {'accuracy': acc}

        

        trainer = WeightedTrainer(

            class_weights=self.class_weights,

            model=self.model,

            args=training_args,

            train_dataset=train_dataset,

            eval_dataset=val_dataset,

            compute_metrics=compute_metrics,

            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]

        )

        

        print("\nStarting training with class balancing...")

        trainer.train()

        

        return trainer

    

    def evaluate(self, test_dataset, label_mapping=None):

        """Evaluate with confusion matrix"""

        

        self.model.eval()

        predictions = []

        true_labels = []

        

        for i in range(len(test_dataset)):

            item = test_dataset[i]

            input_ids = item['input_ids'].unsqueeze(0).to(self.device)

            attention_mask = item['attention_mask'].unsqueeze(0).to(self.device)

            

            with torch.no_grad():

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                pred = torch.argmax(outputs.logits, dim=1).cpu().item()

            

            predictions.append(pred)

            true_labels.append(item['labels'].item())

        

        print("\n" + "="*60)

        print("EVALUATION RESULTS")

        print("="*60)

        

        accuracy = accuracy_score(true_labels, predictions)

        print(f"\nOverall Accuracy: {accuracy:.4f}")

        

        if label_mapping:

            reverse_mapping = {v: k for k, v in label_mapping.items()}

            target_names = [reverse_mapping[i] for i in sorted(reverse_mapping.keys())]

        else:

            target_names = [f"Class_{i}" for i in range(self.num_labels)]

        

        print("\nDetailed Classification Report:")

        print(classification_report(true_labels, predictions, target_names=target_names, zero_division=0))

        

        # Confusion Matrix

        print("\nConfusion Matrix:")

        cm = confusion_matrix(true_labels, predictions)

        print("(rows=true, cols=predicted)")

        for i, row in enumerate(cm):

            print(f"{target_names[i]:25s}: {row}")

        

        return accuracy, predictions, true_labels

    

    def predict(self, text, return_all_probs=False):

        """Predict with confidence scores for all classes"""

        

        self.model.eval()

        

        encoding = self.tokenizer.encode_plus(

            text,

            add_special_tokens=True,

            max_length=512,

            padding='max_length',

            truncation=True,

            return_attention_mask=True,

            return_tensors='pt'

        )

        

        input_ids = encoding['input_ids'].to(self.device)

        attention_mask = encoding['attention_mask'].to(self.device)

        

        with torch.no_grad():

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

            pred = torch.argmax(probs, dim=1).cpu().item()

            confidence = probs[0][pred].cpu().item()

            

            if return_all_probs:

                all_probs = probs[0].cpu().numpy()

                return pred, confidence, all_probs

        

        return pred, confidence

    

    def save_model(self, path='./trained_model'):

        """Save the fine-tuned model"""

        self.model.save_pretrained(path)

        self.tokenizer.save_pretrained(path)

        

        # Save class weights

        if self.class_weights is not None:

            torch.save(self.class_weights, f"{path}/class_weights.pt")

        

        print(f"Model saved to {path}")

    

    def load_model(self, path='./trained_model'):

        """Load a saved model"""

        self.model = DistilBertForSequenceClassification.from_pretrained(path)

        self.tokenizer = DistilBertTokenizer.from_pretrained(path)

        self.model.to(self.device)

        

        # Load class weights if available

        weights_path = f"{path}/class_weights.pt"

        if Path(weights_path).exists():

            self.class_weights = torch.load(weights_path)

        

        print(f"Model loaded from {path}")



# ============================================

# MAIN EXECUTION

# ============================================



def main():

    """Main execution pipeline with improvements"""

    

    ROOT_FOLDER = 'D:\capstone\github_repo\Text_Extraction\Extracted_Text'

    MIN_SAMPLES_PER_CLASS = 10

    EPOCHS = 7

    OUTPUT_DIR = './results'

    MODEL_SAVE_PATH = './trained_model'

    

    print("="*60)

    print("IMPROVED TEXT CLASSIFICATION WITH CLASS BALANCING")

    print("="*60)

    print("\n1. Loading and augmenting data...")

    df, label_mapping = load_data_from_folders(

        ROOT_FOLDER,

        min_samples=MIN_SAMPLES_PER_CLASS,

        augment=True

    )

    num_labels = len(label_mapping)  # Use len(label_mapping) instead of nunique

    

    print(f"\nFinal configuration:")

    print(f"  Total samples: {len(df)}")

    print(f"  Number of classes: {num_labels}")

    print(f"  Classes: {list(label_mapping.keys())}")

    

    # Save label mapping

    import json

    with open('label_mapping.json', 'w') as f:

        json.dump(label_mapping, f, indent=2)

    print(f"\nLabel mapping saved to 'label_mapping.json'")

    

    # Initialize classifier with correct number of labels

    print("\n2. Initializing improved classifier...")

    classifier = ImprovedTextClassifier(num_labels=num_labels)

    

    # Prepare datasets

    print("\n3. Preparing datasets with class balancing...")

    train_dataset, val_dataset, test_dataset = classifier.prepare_datasets(

        df, label_mapping

    )

    

    # Train

    print("\n4. Training model with class weights...")

    trainer = classifier.train(train_dataset, val_dataset, OUTPUT_DIR, EPOCHS)

    

    # Evaluate

    print("\n5. Evaluating model...")

    accuracy, predictions, true_labels = classifier.evaluate(test_dataset, label_mapping)

    

    # Save

    print("\n6. Saving model...")

    classifier.save_model(MODEL_SAVE_PATH)

    

    print("\n" + "="*60)

    print("TRAINING COMPLETE!")

    print("="*60)



# ============================================

# PREDICTION WITH TOP-K RESULTS

# ============================================



def predict_with_top_k(text, model_path='./trained_model', label_mapping=None, k=3):

    """

    Predict with top K most likely categories

    """

    import json

    

    if label_mapping is None:

        with open('label_mapping.json', 'r') as f:

            label_mapping = json.load(f)

    classifier = ImprovedTextClassifier(num_labels=len(label_mapping))

    classifier.load_model(model_path)

    

    pred_label, confidence, all_probs = classifier.predict(text, return_all_probs=True)

    

    # Get top K predictions

    top_k_indices = np.argsort(all_probs)[-k:][::-1]

    

    reverse_mapping = {v: k for k, v in label_mapping.items()}

    

    print(f"\nTop {k} predictions:")

    print("-" * 60)

    for i, idx in enumerate(top_k_indices, 1):

        category = reverse_mapping[idx]

        prob = all_probs[idx]

        print(f"{i}. {category:30s}: {prob:.2%}")

    

    return reverse_mapping[pred_label], confidence



if __name__ == "__main__":
    
    
    predict_with_top_k('''[ xHacker ]
    [ My Services ]
    [ Hacking Education ]
    [ Contact Me ]



[ About xHacker ]

I am a independent security researcher. Hacking and social engineering is my business since 2008. I never had a real job so I had the time to get really good at this because I have spent the half of my life studying and researching about hacking, engineering and web technologies. I have worked for other people before in Silk Road and now I'm also offering my services for everyone with enough cash.
Technical Skills

- Web (HTML, PHP, SQL, APACHE).
- C/C++, Java, Javascript, Python.
- 0day Exploits, Highly personalized trojans, Bots, DDOS attacks.
- Spear Phishing Attacks to get passwords from selected targets.
- Hacking Web Technologies (Fuzzing, NO/SQLi, XSS, LDAP, Xpath).
- Social Engineering

[ My Services ]

I am a independent security researcher. Hacking and social engineering is my business since 2004. I never had a real job so I had the time to get really good at this because I have spent the half of my life studying and researching about hacking, engineering and web technologies. I have worked for other people before in Silk Road and now I'm also offering my services for everyone with enough cash.

What I'll do

I'll do anything for money, I'm not a pussy ;)
If you want me for perform some illegal shit or work against government targets, I'll do it!
Some examples:
- Hacking web servers, computers and smartphones.
- Malware development for any operating system.
- Economic espionage and corporate espionage.
- Getting private information from someone.
- Change grades in schools and universities.
- Password recovery.
...and much more!

Prices

I'm not doing this to make a few bucks here and there, I'm not some shit of eastern europe country who is happy to scam people for 50 bucks.
I'm a proffessional computer expert who could earn 50-100 bucks an hour with a legal job. So stop reading if you don't have a serious problem worth spending some cash at. Prices depend a lot of the problem you want me to solve, but the minimum amount for smaller jobs is 200 USD.
You can pay me anonymously using Bitcoin.
Learn more about prices on My Services page
Payment conditions

I won't start any work without the payment as a guarantee.
Here are 2 ways to pay:
- Via bitcoin directly 50% before and 50% upon the completion
- Via Bitcoin 100% before through escrow
Escrow conditions

Escrow protects my time and your money. It is okay to use escrow with me.
Because of security, there are some conditions:
- Clearnet escrow services only.
- Escrow should accept Bitcoin payments.
- Escrow should allow illegal deals. Usually you can find out this in their terms of using page.
- I am not going to offer you escrow service. It is always your choice. You choose the exact escrow service you are going to use.
- I am not going to check if the escrow allows illegal deals. Please do it on your own.

xhacker@safe-mail.net2015-2025 © Xhacker
''', k=5)