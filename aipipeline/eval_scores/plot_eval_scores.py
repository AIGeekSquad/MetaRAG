# Description: This file contains the WIP implementation to plot evaluation values or points of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_model_comparison(self, answer_columns=["generated_answer", "ft_generated_answer"], scenario="answer_expected", nice_names=["Baseline", "Fine-Tuned"]):
        """
        Plots a comparison of model evaluation results based on the specified scenario.
        Parameters:
        - answer_columns (list of str): List of column names containing the answers to be evaluated.
        - scenario (str): The scenario to be used for evaluation. It can be either "answer_expected" or "idk_expected".
        - nice_names (list of str): List of names to be used for labeling the models in the plot.
        Raises:
        - ValueError: If an invalid scenario is provided.
        The function evaluates the models based on the specified scenario, creates a DataFrame of the results,
        and generates a bar plot comparing the models. The plot is annotated with the percentage values for each bar.
        """
        
        results = []
        for col in answer_columns:
            answer_expected, idk_expected = self.evaluate_model(col)
            if scenario == "answer_expected":
                results.append(answer_expected)
            elif scenario == "idk_expected":
                results.append(idk_expected)
            else:
                raise ValueError("Invalid scenario")
        
        
        results_df = pd.DataFrame(results, index=nice_names)
        if scenario == "answer_expected":
            results_df = results_df.reindex(self.labels_answer_expected, axis=1)
        elif scenario == "idk_expected":
            results_df = results_df.reindex(self.labels_idk_expected, axis=1)
        
        melted_df = results_df.reset_index().melt(id_vars='index', var_name='Status', value_name='Frequency')
        sns.set_theme(style="whitegrid", palette="icefire")
        g = sns.catplot(data=melted_df, x='Frequency', y='index', hue='Status', kind='bar', height=5, aspect=2)

        # Annotating each bar
        for p in g.ax.patches:
            g.ax.annotate(f"{p.get_width():.0f}%", (p.get_width()+5, p.get_y() + p.get_height() / 2),
                        textcoords="offset points",
                        xytext=(0, 0),
                        ha='center', va='center')
        plt.ylabel("Model")
        plt.xlabel("Percentage")
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.title(scenario.replace("_", " ").title())
        plt.show()


        # Compare the results by merging into one dataframe
        ##evaluator = Evaluator(df)