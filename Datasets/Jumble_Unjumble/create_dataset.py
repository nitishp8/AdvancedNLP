import pandas as pd
import random

# Function to jumble a sentence
def jumble_sentence(sentence):
    words = sentence.split()
    random.shuffle(words)
    return ' '.join(words)

# New article lines
article = [
    "The benefits of regular exercise are well-documented.",
    "Physical activity boosts mental and physical health.",
    "Exercise helps manage weight and prevent obesity.",
    "Cardiovascular workouts improve heart health.",
    "Strength training builds muscle and bone density.",
    "Flexibility exercises enhance range of motion.",
    "Regular exercise can reduce the risk of chronic diseases.",
    "Physical activity improves mood and reduces stress.",
    "Exercise promotes better sleep and relaxation.",
    "Staying active increases energy levels.",
    "Exercise helps maintain a healthy immune system.",
    "Physical activity can improve cognitive function.",
    "Exercise is important for children’s development.",
    "Group fitness classes provide motivation and social interaction.",
    "Outdoor activities connect you with nature.",
    "Setting fitness goals can enhance your motivation.",
    "Tracking progress helps maintain fitness routines.",
    "Warming up and cooling down prevent injuries.",
    "Hydration is crucial during exercise.",
    "Eating a balanced diet complements your fitness efforts.",
    "Regular exercise supports mental well-being.",
    "Exercise releases endorphins, boosting happiness.",
    "Fitness can be fun with diverse activities.",
    "Consistency is key to achieving fitness goals.",
    "Finding a workout buddy can increase accountability.",
    "Rest days are important for recovery.",
    "Listening to music can enhance workout performance.",
    "Exercise can be adapted for all fitness levels.",
    "Maintaining proper form prevents injuries.",
    "Physical activity supports a healthy metabolism.",
    "Exercise can improve self-esteem and confidence.",
    "Morning workouts can boost your daily productivity.",
    "Fitness trackers can monitor your activity levels.",
    "Joining a sports team can provide structured exercise.",
    "Exercise can be a great stress reliever.",
    "Dancing is a fun way to stay active.",
    "Yoga combines physical activity with mindfulness.",
    "Exercise can be a part of your daily routine.",
    "Regular activity helps maintain a healthy weight.",
    "Exercise can reduce the risk of mental health issues.",
    "Physical activity can be a social event.",
    "Exercise improves flexibility and balance.",
    "Engaging in sports can build teamwork skills.",
    "Exercise can improve cardiovascular endurance.",
    "Staying active promotes longevity.",
    "Fitness challenges can motivate you to stay active.",
    "Variety in workouts prevents boredom.",
    "Exercise can help manage chronic pain.",
    "Physical activity is important at all ages.",
    "Exercise can enhance your overall quality of life."
]

# Create the dataset
data = {
    "jumbled_sentences": [jumble_sentence(line) for line in article],
    "unjumbled_sentences": article
}

# Create DataFrame
df = pd.DataFrame(data)

# Split into train and test datasets
train_df = df.iloc[:40]
test_df = df.iloc[40:]

# Save to .tsv files
train_df.to_csv('train10.tsv', sep='\t', index=False)
test_df.to_csv('test10.tsv', sep='\t', index=False)

print("Train and test datasets created successfully!")
