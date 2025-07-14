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


# ========================================================================
#   Load env variables
# ========================================================================

# Load environment variables
load_dotenv()
warnings.filterwarnings("ignore")


# ========================================================================
#   Helper functions
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

    # Instantiate sentence transformer model
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
        # Include bigrams for better context
        ngram_range=(1, 2),
    )

    # Instantiate BERTopic model
    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        # Limit: 10 topics
        nr_topics=10,
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

    # -1 to exclude outlier topic
    print(f"\n✓ Found {len(topic_info) - 1} topics!")

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
    prompt = f"""
        You are analyzing topics from product reviews. Based on the following information, provide a clear interpretation.

        Top words/phrases for Topic {topic_num}:
        {', '.join(topic_words[:15])}

        Example reviews (excerpts):
        1. "{example_docs[0][:300]}..."
        2. "{example_docs[1][:300]}..."
        3. "{example_docs[2][:300]}..."

        Please provide:
        1. A concise topic name (2–5 words) that describes the product category being reviewed (e.g., 'Streaming Devices', 'E-Readers', 'Headphones'). 
        Avoid words like 'reviews' or 'issues' and just list the product category. Also avoid overly broad topics like 'Electronic Devices'.
        2. The main themes discussed in this topic (bullet points).
        3. The type of product(s) these reviews are likely about.

        Frame your interpretation around product types/categories rather than specific sentiment or brand names.

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
                {"role": "user", 
                 "content": prompt
                },
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


def analyze_all_topics_with_gpt(topic_model, reviews, topics, max_topics=15):
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

    # ================================================
    #   Topic distribution bar chart
    # ================================================
    # Generate count of topics
    topic_counts = pd.Series(topics).value_counts()
    
    # Get list of valid topic numbers (0 to n), excluding -1 outlier topic if present
    topic_counts = topic_counts[topic_counts.index != -1]  

    # Add interpreted names to the chart
    # Sort in descending order for horizontal display
    topic_counts_sorted = topic_counts.sort_values(ascending=True)  
    
    # Reorder topic names to match
    topic_names_sorted = []
    for topic_num in topic_counts_sorted.index:
        if topic_num in interpretations:
            topic_names_sorted.append(f"{interpretations[topic_num]['name']}")
        else:
            topic_names_sorted.append(f"Topic {topic_num}")
    
    fig_dist = go.Figure(
        data=[
            go.Bar(
                x=topic_counts_sorted.values,           # Use sorted values
                y=topic_names_sorted,                   # Use sorted names          
                orientation="h",                        # Horizontal                                     
                text=topic_counts_sorted.values,        # Use count for data labels
                textposition="outside",                 # Place data labels outside bars
                marker_color="#FFA500",               # Annielytics orange
                marker=dict(color="#FFA500", 
                            line=dict(color="#FFA500",
                                      width=0)),
            )
        ]
    )

    fig_dist.update_layout(
        title={
            'text': "Topic Distribution in Reviews",
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=18)
        },
        yaxis_tickangle=0,                                                  # No angle needed for horizontal
        xaxis_tickangle=0,
        height=500,
        plot_bgcolor="white",                                               # White background
        paper_bgcolor="white",
        margin_pad=10,
        xaxis=dict(
            showgrid=False,                                                 # No gridlines
            zeroline=True, 
            showticklabels=False),                                          
        yaxis=dict(
            showgrid=False,                                                 # No gridlines
            zeroline=True)                                          
    )

    fig_dist.write_html("output/topic_distribution_interactive.html")
    print("  ✓ Saved interactive topic distribution")
    
    
    # ================================================
    #   Topic similarity heatmap 
    # ================================================

    topic_embeddings = topic_model._extract_embeddings(
        [
            " ".join([word for word, _ in topic_model.get_topic(t)])
            for t in range(len(set(topics)) - 1)
        ]
    )

    # Calculate similarity matrix using cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity

    similarity_matrix = cosine_similarity(topic_embeddings)

    # Create heatmap with topic names
    heatmap_labels = []
    for i in range(len(similarity_matrix)):
        if i in interpretations:
            heatmap_labels.append(interpretations[i]["name"])
        else:
            heatmap_labels.append(f"Topic {i}")

    # Convert to percentages for display
    similarity_percentages = np.round(similarity_matrix * 100).astype(int)
    
    # Convert to percentages for display
    similarity_percentages = np.round(similarity_matrix * 100).astype(int)

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=similarity_matrix,
            x=heatmap_labels,
            y=heatmap_labels,
            colorscale=[
                [0, "#FFF6E6"],
                [1, "#FFA500"],
            ],  # My brand colors
            text=similarity_percentages,                                        # Show percentages
            texttemplate="%{text}%",                                            # Add % symbol
            textfont={"size": 10},
            hovertemplate="%{x} - %{y}<br>Similarity: %{text}%<extra></extra>", # Better hover
        )
    )

    fig_heat.update_layout(
        title={
            'text': "Topic Similarity Matrix",
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=18)
        },
        height=600, 
        xaxis_tickangle=-45,       
        margin_pad=10,                                                           # Air out yaxis labels
        paper_bgcolor='white',                                                   # Remove gray border around heatmap
        plot_bgcolor='white'                                                     # Remove gray border around heatmap
    )

    fig_heat.write_html("output/topic_similarity_interactive.html")
    print("  ✓ Saved interactive topic similarity matrix")


    # ================================================
    #   Topic bubble chart from BERTopic
    # ================================================

    # Calculate topic similarity using embeddings
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Get topic words for similarity calculation
    topic_words_list = []
    valid_topics = [t for t in set(topics) if t != -1]
    valid_topics.sort()
    
    for topic_num in valid_topics:
        words = [word for word, _ in topic_model.get_topic(topic_num)][:20]
        topic_words_list.append(" ".join(words))
    
    # Create embeddings and calculate similarity
    from sentence_transformers import SentenceTransformer
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_embeddings = sentence_model.encode(topic_words_list)
    similarity_matrix = cosine_similarity(topic_embeddings)
    
    # Use TSNE for 2D positioning based on embeddings directly
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    topic_positions = tsne.fit_transform(topic_embeddings)
    
    # Normalize positions to use full viewport
    x_min, x_max = topic_positions[:, 0].min(), topic_positions[:, 0].max()
    y_min, y_max = topic_positions[:, 1].min(), topic_positions[:, 1].max()
    
    # Scale to use more of the viewport
    x_range = x_max - x_min
    y_range = y_max - y_min
    topic_positions[:, 0] = ((topic_positions[:, 0] - x_min) / x_range - 0.5) * 10
    topic_positions[:, 1] = ((topic_positions[:, 1] - y_min) / y_range - 0.5) * 8
    
    # Get topic data
    topic_data = []
    for i, topic_num in enumerate(valid_topics):
        size = len([t for t in topics if t == topic_num])
        
        if topic_num in interpretations:
            name = interpretations[topic_num]['name']
            desc = interpretations[topic_num]['description']
        else:
            name = f"Topic {topic_num + 1}"
            desc = "No interpretation available"
        
        topic_data.append({
            'topic_num': topic_num,
            'display_num': topic_num + 1,                   # Start with 1 instead of 0
            'name': name,
            'description': desc,
            'size': size,
            'x': topic_positions[i, 0],
            'y': topic_positions[i, 1],
            'percentage': (size / len(topics)) * 100
        })
    
    # Create the visualization
    fig_topics_custom = go.Figure()
    
    # Add connection lines for highly similar topics
    for i in range(len(valid_topics)):
        for j in range(i + 1, len(valid_topics)):
            if similarity_matrix[i, j] > 0.7:  # High similarity threshold
                fig_topics_custom.add_trace(go.Scatter(
                    x=[topic_data[i]['x'], topic_data[j]['x']],
                    y=[topic_data[i]['y'], topic_data[j]['y']],
                    mode='lines',
                    line=dict(color='rgba(128, 128, 128, 0.3)', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Add bubbles
    for i, topic in enumerate(topic_data):
        # Color based on size (larger = darker orange)
        opacity = 0.4 + (topic['percentage'] / 100) * 0.6
        color = '#FFA500'  # Use your brand orange
        
        bubble_size = 30 + np.sqrt(topic['size']) * 10
        
        fig_topics_custom.add_trace(go.Scatter(
            x=[topic['x']],
            y=[topic['y']],
            mode='markers+text',
            marker=dict(
                size=bubble_size,
                color=color,
                opacity=0.8,
                line=dict(color='white', width=2)
            ),
            text=f"<b>{topic['name']}</b>",
            textposition="middle center",
            textfont=dict(size=11),
            hovertemplate=(
                f"<b>{topic['name']}</b><br>" +
                f"Topic {topic['display_num']}<br>" +
                f"Size: {topic['size']} reviews ({topic['percentage']:.1f}%)<br>" +
                f"<i>{topic['description']}</i>" +
                "<extra></extra>"
            ),
            hoverlabel= dict(bgcolor='rgba(139, 180, 45, 0.7)'),
            showlegend=False
        ))
    
    # Full viewport layout
    fig_topics_custom.update_layout(
        title={
            'text': "Topic Similarity Map",
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=18)
        },
        xaxis=dict(
            visible=True,
            range=[-9, 9],                                      # Wider range to prevent cutoff
            showgrid=False,
            zeroline=False,
            zerolinecolor='lightgray',
            showline=True,                                      # Show axis line
            linewidth=2,                                        # Make it visible
            linecolor='#DEDEDE',                              # Gray border
            mirror=True,                                        # Show on all sides
            showticklabels=False,
            title=""
        ),
        yaxis=dict(
            visible=True,
            range=[-7, 7],                                      # Wider range to prevent cutoff  
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=True,                                      # Show axis line
            linewidth=2,                                        # Make it visible
            linecolor='#DEDEDE',                              # Gray border
            mirror=True,                                        # Show on all sides
            title=""
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=800,
        width=1200,
        margin=dict(l=100, r=100, t=80, b=80),                  # Bigger margins
        hovermode='closest'
    )
    
    fig_topics_custom.write_html("output/topic_clusters_interactive.html")
    print("  ✓ Saved interactive topic clusters")
    

def save_detailed_results(
    topic_model,
    topics,
    interpretations,
    reviews,
    output_file="output/topic_analysis_results.csv",
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

    # Create output directory if it doesn't exist
    import os

    os.makedirs("output", exist_ok=True)

    # Configuration
    filepath = "data/product-reviews-kaggle.csv"

    # Load the data
    reviews, df = load_reviews(filepath)

    # Create BERTopic model
    topic_model, sentence_model = create_bertopic_model()

    # Run topic modeling
    topics, probs, topic_info = run_topic_modeling(topic_model, reviews)

    # Display basic topic info
    print("\n📊 Topic Overview:")
    print(topic_info.head(10))

    # Use GPT to interpret topics
    interpretations = analyze_all_topics_with_gpt(
        topic_model, reviews, topics, max_topics=10  # Analyze top 10 topics
    )

    # Create visualizations
    create_interactive_visualizations(topic_model, topics, probs, interpretations)

    # Save detailed results
    save_detailed_results(topic_model, topics, interpretations, reviews)

    # Create and save summary report
    report = create_topic_summary_report(interpretations, topic_info)

    with open("output/topic_analysis_report.txt", "w") as f:
        f.write(report)

    print("\n" + report)

    print("\n🎉 Analysis Complete!")
    print("\nYou now have:")
    print(
        "  ✓ output/topic_distribution_interactive.html - Interactive topic distribution"
    )
    print("  ✓ output/topic_similarity_interactive.html - Topic similarity matrix")
    print("  ✓ output/topic_clusters_interactive.html - Topic cluster visualization")
    print("  ✓ output/topic_analysis_results.csv - Detailed results with all reviews")
    print("  ✓ output/topic_analysis_report.txt - Summary report")

    return topic_model, topics, interpretations

# ========================================================================
#   Process csv
# ========================================================================

if __name__ == "__main__":
    # If API key isn't set
    if not openai.api_key:
        print("⚠️  Please set your OpenAI API key first!")
        print("You can do this by:")
        print("1. Setting an environment variable: export OPENAI_API_KEY='your-key'")
        print("2. Or directly in the script: openai.api_key = 'your-key'")
    else:
        topic_model, topics, interpretations = main()