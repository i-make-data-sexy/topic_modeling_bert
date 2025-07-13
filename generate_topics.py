# ========================================================================
#   Import libraries
# ========================================================================

import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import openai
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go
import plotly.express as px
from umap import UMAP
import warnings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
warnings.filterwarnings("ignore")


# ========================================================================
#   Process
# ========================================================================


def load_reviews(filepath):
    """
    Loads the CSV file and extracts review texts.

    Args:
        filepath (str): Path to the CSV file containing reviews

    Returns:
        tuple: (reviews list, full dataframe)
    """
    print("📚 Loading reviews from CSV...")

    # Load the data
    df = pd.read_csv(filepath)

    # Extract reviews and remove any NaN values
    reviews = df["reviews.text"].dropna().tolist()

    print(f"✓ Loaded {len(reviews)} reviews successfully!")

    return reviews, df


def create_bertopic_model():
    """
    Creates and configures a BERTopic model with custom settings.

    Args:
        None

    Returns:
        tuple: (BERTopic model, sentence transformer model)
    """
    print("\n🤖 Setting up BERTopic model...")

    # Use a good sentence transformer model
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Configure UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
    )

    # Custom stopwords for product reviews
    stop_words = [
        "product",
        "amazon",
        "item",
        "one",
        "would",
        "get",
        "got",
        "like",
        "really",
        "much",
        "also",
        "well",
        "even",
        "still",
        "just",
        "dont",
        "ive",
        "im",
        "thing",
        "things",
        "way",
        "time",
        "use",
        "used",
        "using",
        "review",
        "bought",
        "purchase",
    ]

    # Configure CountVectorizer for better terms
    vectorizer_model = CountVectorizer(
        stop_words=stop_words,
        min_df=5,
        ngram_range=(1, 2),  # Include bigrams for better context
    )

    # Create BERTopic model
    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        verbose=True,
    )

    print("✓ Model configured successfully!")

    return topic_model, sentence_model


def run_topic_modeling(topic_model, reviews):
    """
    Runs BERTopic on the reviews to discover topics.

    Args:
        topic_model (BERTopic): Configured BERTopic model
        reviews (list): List of review texts

    Returns:
        tuple: (topics, probabilities, topic_info)
    """
    print("\n🔍 Discovering topics (this may take a few minutes)...")

    # Fit the model and transform the reviews
    topics, probs = topic_model.fit_transform(reviews)

    # Get topic information
    topic_info = topic_model.get_topic_info()

    print(f"\n✓ Found {len(topic_info) - 1} topics!")  # -1 to exclude outlier topic

    return topics, probs, topic_info


def get_gpt_interpretation(topic_words, example_docs, topic_num):
    """
    Uses GPT to interpret and name a topic based on its words and example documents.

    Args:
        topic_words (list): Top words for the topic
        example_docs (list): Example documents for the topic
        topic_num (int): Topic number

    Returns:
        dict: GPT's interpretation including name and analysis
    """
    # Prepare the prompt
    prompt = f"""You are analyzing topics from product reviews. Based on the following information, provide a clear interpretation:

Top words/phrases for Topic {topic_num}:
{', '.join(topic_words[:15])}

Example reviews (excerpts):
1. "{example_docs[0][:300]}..."
2. "{example_docs[1][:300]}..."
3. "{example_docs[2][:300]}..."

Please provide:
1. A concise topic name (2-5 words)
2. A brief description (1-2 sentences) of what this topic represents
3. The main themes discussed in this topic (bullet points)
4. What type of product(s) these reviews are likely about

Format your response as JSON with keys: "name", "description", "themes", "product_type"
"""

    try:
        # Call GPT-4
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing and interpreting topic modeling results.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )

        # Parse the response
        import json

        result = json.loads(response.choices[0].message.content)

        return result

    except Exception as e:
        print(f"  ⚠️  GPT interpretation failed for Topic {topic_num}: {str(e)}")

        # Return a basic interpretation if GPT fails
        return {
            "name": f"Topic {topic_num}",
            "description": "Manual interpretation needed",
            "themes": topic_words[:5],
            "product_type": "Unknown",
        }


def analyze_all_topics_with_gpt(topic_model, reviews, topics, max_topics=10):
    """
    Analyzes all discovered topics using GPT for interpretation.

    Args:
        topic_model (BERTopic): Fitted BERTopic model
        reviews (list): Original reviews
        topics (list): Topic assignments for each review
        max_topics (int): Maximum number of topics to analyze

    Returns:
        dict: Dictionary of GPT interpretations for each topic
    """
    print("\n🧠 Using GPT to interpret topics...")

    interpretations = {}

    # Get unique topics (excluding -1 which is outliers)
    unique_topics = [t for t in set(topics) if t != -1]
    unique_topics.sort()

    # Limit to max_topics
    topics_to_analyze = unique_topics[:max_topics]

    for topic_num in topics_to_analyze:
        print(f"\n  Analyzing Topic {topic_num}...")

        # Get topic words
        topic_words = [word for word, _ in topic_model.get_topic(topic_num)]

        # Get example documents for this topic
        topic_docs_indices = [i for i, t in enumerate(topics) if t == topic_num]
        example_indices = np.random.choice(
            topic_docs_indices, size=min(3, len(topic_docs_indices)), replace=False
        )
        example_docs = [reviews[i] for i in example_indices]

        # Get GPT interpretation
        interpretation = get_gpt_interpretation(topic_words, example_docs, topic_num)
        interpretations[topic_num] = interpretation

        # Display the interpretation
        print(f"  ✓ Topic {topic_num}: {interpretation['name']}")
        print(f"    Description: {interpretation['description']}")

    return interpretations


def create_interactive_visualizations(topic_model, topics, probs, interpretations):
    """
    Creates interactive visualizations using Plotly.

    Args:
        topic_model (BERTopic): Fitted BERTopic model
        topics (list): Topic assignments
        probs (array): Topic probabilities
        interpretations (dict): GPT interpretations

    Returns:
        None
    """
    print("\n📊 Creating interactive visualizations...")

    # 1. Topic distribution bar chart
    topic_counts = pd.Series(topics).value_counts()
    topic_counts = topic_counts[topic_counts.index != -1]  # Remove outliers

    # Add interpreted names to the chart
    topic_names = []
    for topic_num in topic_counts.index:
        if topic_num in interpretations:
            topic_names.append(f"{interpretations[topic_num]['name']}")
        else:
            topic_names.append(f"Topic {topic_num}")

    fig_dist = go.Figure(
        data=[
            go.Bar(
                x=topic_names,
                y=topic_counts.values,
                text=topic_counts.values,
                textposition="auto",
                marker_color="lightblue",
            )
        ]
    )

    fig_dist.update_layout(
        title="Topic Distribution in Reviews",
        xaxis_title="Topics",
        yaxis_title="Number of Reviews",
        xaxis_tickangle=-45,
        height=500,
    )

    fig_dist.write_html("topic_distribution_interactive.html")
    print("  ✓ Saved interactive topic distribution")

    # 2. Topic similarity heatmap
    topic_embeddings = topic_model._extract_embeddings(
        [
            " ".join([word for word, _ in topic_model.get_topic(t)])
            for t in range(len(set(topics)) - 1)
        ]
    )

    # Calculate similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity

    similarity_matrix = cosine_similarity(topic_embeddings)

    # Create heatmap with topic names
    heatmap_labels = []
    for i in range(len(similarity_matrix)):
        if i in interpretations:
            heatmap_labels.append(interpretations[i]["name"])
        else:
            heatmap_labels.append(f"Topic {i}")

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=similarity_matrix,
            x=heatmap_labels,
            y=heatmap_labels,
            colorscale="Viridis",
            text=np.round(similarity_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )

    fig_heat.update_layout(
        title="Topic Similarity Matrix", height=600, xaxis_tickangle=-45
    )

    fig_heat.write_html("topic_similarity_interactive.html")
    print("  ✓ Saved interactive topic similarity matrix")

    # 3. Save topic visualization from BERTopic
    fig_topics = topic_model.visualize_topics()
    fig_topics.write_html("topic_clusters_interactive.html")
    print("  ✓ Saved interactive topic clusters")

    print("\n✓ All visualizations saved!")


def save_detailed_results(
    topic_model,
    topics,
    interpretations,
    reviews,
    output_file="topic_analysis_results.csv",
):
    """
    Saves detailed results including topic assignments and interpretations.

    Args:
        topic_model (BERTopic): Fitted BERTopic model
        topics (list): Topic assignments
        interpretations (dict): GPT interpretations
        reviews (list): Original reviews
        output_file (str): Output filename

    Returns:
        None
    """
    print(f"\n💾 Saving detailed results to {output_file}...")

    # Create a detailed results dataframe
    results_data = []

    for i, (review, topic) in enumerate(zip(reviews, topics)):
        if topic == -1:
            topic_name = "Outlier/No Clear Topic"
            topic_desc = "Review doesn't fit clearly into any topic"
        elif topic in interpretations:
            topic_name = interpretations[topic]["name"]
            topic_desc = interpretations[topic]["description"]
        else:
            topic_name = f"Topic {topic}"
            topic_desc = "No interpretation available"

        results_data.append(
            {
                "review_id": i,
                "review_text": review[:500] + "..." if len(review) > 500 else review,
                "topic_number": topic,
                "topic_name": topic_name,
                "topic_description": topic_desc,
            }
        )

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_file, index=False)

    print(f"✓ Saved {len(results_df)} reviews with topic assignments")


def create_topic_summary_report(interpretations, topic_info):
    """
    Creates a readable summary report of all topics.

    Args:
        interpretations (dict): GPT interpretations
        topic_info (DataFrame): BERTopic topic information

    Returns:
        str: Formatted report
    """
    report = "=" * 60 + "\n"
    report += "TOPIC ANALYSIS SUMMARY REPORT\n"
    report += "=" * 60 + "\n\n"

    for topic_num, interp in interpretations.items():
        # Get topic size
        topic_size = topic_info[topic_info["Topic"] == topic_num]["Count"].values[0]

        report += f"📌 TOPIC {topic_num}: {interp['name'].upper()}\n"
        report += "-" * 40 + "\n"
        report += f"Size: {topic_size} reviews\n"
        report += f"Description: {interp['description']}\n"
        report += f"Product Type: {interp['product_type']}\n"
        report += "\nMain Themes:\n"

        if isinstance(interp["themes"], list):
            for theme in interp["themes"]:
                report += f"  • {theme}\n"
        else:
            report += f"  • {interp['themes']}\n"

        report += "\n"

    return report


# Main execution function
def main():
    """
    Main function to run the complete BERTopic + GPT analysis pipeline.

    Args:
        None

    Returns:
        None
    """
    print("🚀 Starting Modern Topic Analysis with BERTopic and GPT!")
    print("=" * 60)

    # Configuration
    filepath = "productreviewskaggle.csv"

    # Step 1: Load the data
    reviews, df = load_reviews(filepath)

    # Step 2: Create BERTopic model
    topic_model, sentence_model = create_bertopic_model()

    # Step 3: Run topic modeling
    topics, probs, topic_info = run_topic_modeling(topic_model, reviews)

    # Display basic topic info
    print("\n📊 Topic Overview:")
    print(topic_info.head(10))

    # Step 4: Use GPT to interpret topics
    interpretations = analyze_all_topics_with_gpt(
        topic_model, reviews, topics, max_topics=10  # Analyze top 10 topics
    )

    # Step 5: Create visualizations
    create_interactive_visualizations(topic_model, topics, probs, interpretations)

    # Step 6: Save detailed results
    save_detailed_results(topic_model, topics, interpretations, reviews)

    # Step 7: Create and save summary report
    report = create_topic_summary_report(interpretations, topic_info)

    with open("topic_analysis_report.txt", "w") as f:
        f.write(report)

    print("\n" + report)

    print("\n🎉 Analysis Complete!")
    print("\nYou now have:")
    print("  ✓ topic_distribution_interactive.html - Interactive topic distribution")
    print("  ✓ topic_similarity_interactive.html - Topic similarity matrix")
    print("  ✓ topic_clusters_interactive.html - Topic cluster visualization")
    print("  ✓ topic_analysis_results.csv - Detailed results with all reviews")
    print("  ✓ topic_analysis_report.txt - Summary report")

    return topic_model, topics, interpretations


# Run the analysis
if __name__ == "__main__":
    # Make sure to set your OpenAI API key before running!
    if not openai.api_key:
        print("⚠️  Please set your OpenAI API key first!")
        print("You can do this by:")
        print("1. Setting an environment variable: export OPENAI_API_KEY='your-key'")
        print("2. Or directly in the script: openai.api_key = 'your-key'")
    else:
        topic_model, topics, interpretations = main()
