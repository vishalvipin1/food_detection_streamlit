import pandas as pd
import numpy as np
from ast import literal_eval
from collections import Counter
from tqdm.auto import tqdm
import warnings

from surprise import Reader, Dataset, NMF, accuracy
from surprise.model_selection import train_test_split

class FoodRecommendationSystem:
    def __init__(self, recommended_csv, ratings_csv):
        self.food_df = pd.read_csv(recommended_csv)
        self.ratings_df = pd.read_csv(ratings_csv)
        self.model = None
        self.trainset = None
        self.testset = None
        self._setup()

    def _setup(self):
        # Set pandas display options
        pd.set_option('display.max_colwidth', 1000)
        warnings.filterwarnings("ignore")
        tqdm.pandas()

        # Initialize surprise Reader
        reader = Reader(rating_scale=(0, 5))
        data = Dataset.load_from_df(self.ratings_df[['foodnumber', 'User', 'Rating']], reader)
        self.trainset, self.testset = train_test_split(data, test_size=0.25)

        # Non-negative Matrix Factorization
        self.model = NMF()
        self.model.fit(self.trainset)

        # Evaluate the model
        predictions = self.model.test(self.testset)
        mse = accuracy.mse(predictions)
        rmse = accuracy.rmse(predictions)

        # Count tags
        tags_count = Counter()
        self.food_df["tags"].progress_apply(lambda tags: tags_count.update(literal_eval(tags)))

        TIME_TAGS = [
            '15-minutes-or-less',
            '30-minutes-or-less',
            '60-minutes-or-less',
            '4-hours-or-less',
        ]
        VEGAN_TAGS = ['vegan']
        MEAT_TAGS = [
            'beef',
            'chicken',
        ]

        FEATURE_COLS = TIME_TAGS + VEGAN_TAGS + MEAT_TAGS

        # Function to extract feature tags
        def fe_tags(food_tags):
            values = []

            for group_tag in [TIME_TAGS, VEGAN_TAGS]:
                for tag in group_tag:
                    values.append(True) if tag in food_tags else values.append(False)

            for tag in MEAT_TAGS:
                values.append(True) if tag in food_tags else values.append(False)

            return values   

        # Apply feature tags to dataframe
        self.food_df['tmp'] = self.food_df["tags"].progress_apply(lambda food_tags: fe_tags(food_tags) if food_tags else [False]*len(FEATURE_COLS))
        self.food_df[FEATURE_COLS] = self.food_df['tmp'].apply(pd.Series)
        self.food_df.drop(columns='tmp', inplace=True)

        # Assign time values based on tags
        conds = [
            (self.food_df['4-hours-or-less']),
            (self.food_df['60-minutes-or-less']),
            (self.food_df['30-minutes-or-less']),
            (self.food_df['15-minutes-or-less']),
        ]
        choices = [4, 3, 2, 1]
        self.food_df['time'] = np.select(conds, choices, default=5)

    def recommend_meal(self, uid, filtered_ids, topk):
        preds = []
        for iid in filtered_ids:
            pred_rating = self.model.predict(uid=uid, iid=iid).est
            preds.append([iid, pred_rating])
        preds.sort(key=lambda x: x[1], reverse=True)
        
        return preds[:topk]

    def recommend_vegan(self):
        filtered_ids = self.food_df[(self.food_df['vegan'])]['foodnumber'].to_list()
        random_user = self.ratings_df['User'].sample(1).values[0]
        filtered_df = self.food_df[self.food_df['recommended'] == 0]
        preds = self.recommend_meal(random_user, filtered_ids, 10)
        selected_columns = ['Name', 'Calories', 'SaturatedFatContent', 'CholesterolContent', 'FiberContent']
        recommended_df = filtered_df[filtered_df['foodnumber'].isin([x[0] for x in preds])]
        if recommended_df.empty:
            return "No vegan meals recommended."
        else:
            return recommended_df[selected_columns]

    def recommend_chicken(self):
        filtered_ids = self.food_df[(self.food_df['chicken'])]['foodnumber'].to_list()
        random_user = self.ratings_df['User'].sample(1).values[0]
        filtered_df = self.food_df[self.food_df['recommended'] == 0]
        preds = self.recommend_meal(random_user, filtered_ids, 10)
        recommended_df = filtered_df[filtered_df['foodnumber'].isin([x[0] for x in preds])]
        if recommended_df.empty:
            return "No chicken meals recommended."
        else:
            return recommended_df[['Name','Calories','SaturatedFatContent','CholesterolContent','FiberContent']]

    def recommend_less_time(self):
        filtered_ids = self.food_df[(self.food_df['time']<=1)]['foodnumber'].to_list()
        random_user = self.ratings_df['User'].sample(1).values[0]
        filtered_df = self.food_df[self.food_df['recommended'] == 0]
        preds = self.recommend_meal(random_user, filtered_ids, 10)
        recommended_df = filtered_df[filtered_df['foodnumber'].isin([x[0] for x in preds])]
        if recommended_df.empty:
            return "No meals with less time to make recommended."
        else:
            return recommended_df[['Name','Calories','SaturatedFatContent','CholesterolContent','FiberContent']]

