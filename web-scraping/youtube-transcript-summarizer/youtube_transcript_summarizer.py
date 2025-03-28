import os
import re
import nltk
import pytube
import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from heapq import nlargest
from urllib.parse import urlparse, parse_qs
import textwrap
from colorama import Fore, Back, Style, init
from openai import OpenAI

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Download necessary NLTK data
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="<api_key>", # Add your OpenRouter API key here
)

def extract_video_id(youtube_url):
    """Extract the video ID from a YouTube URL."""
    parsed_url = urlparse(youtube_url)
    
    if parsed_url.netloc == 'youtu.be':
        return parsed_url.path[1:]
    
    if parsed_url.netloc in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
        elif parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]
        elif parsed_url.path.startswith('/v/'):
            return parsed_url.path.split('/')[2]
    
    # If no match found
    raise ValueError(f"Could not extract video ID from URL: {youtube_url}")

def get_transcript(video_id):
    """Get the transcript of a YouTube video."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([entry['text'] for entry in transcript])
    except Exception as e:
        return f"Error retrieving transcript: {str(e)}."

def summarize_text_nltk(text, num_sentences=5):
    """Summarize text using frequency-based extractive summarization with NLTK."""
    if not text or text.startswith("Error") or text.startswith("Transcript not available"):
        return text
    
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    
    # If there are fewer sentences than requested, return all sentences
    if len(sentences) <= num_sentences:
        return text
    
    # Tokenize words and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Calculate word frequencies
    freq = FreqDist(words)
    
    # Score sentences based on word frequencies
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                if i in sentence_scores:
                    sentence_scores[i] += freq[word]
                else:
                    sentence_scores[i] = freq[word]
    
    # Get the top N sentences with highest scores
    summary_sentences_indices = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary_sentences_indices.sort()  # Sort to maintain original order
    
    # Construct the summary
    summary = ' '.join([sentences[i] for i in summary_sentences_indices])
    return summary

def summarize_text_ai(text, video_title, num_sentences=5):
    """Summarize text using the Mistral AI model via OpenRouter."""
    if not text or text.startswith("Error") or text.startswith("Transcript not available"):
        return text
    
    # Truncate text if it's too long (models often have token limits)
    max_chars = 15000  # Adjust based on model's context window
    truncated_text = text[:max_chars] if len(text) > max_chars else text
    
    prompt = f"""Please provide a concise summary of the following YouTube video transcript.
Title: {video_title}

Transcript:
{truncated_text}

Create a clear, informative summary that captures the main points and key insights from the video.
Your summary should be approximately {num_sentences} sentences long.
"""
    
    try:
        completion = client.chat.completions.create(
            model="mistralai/mistral-small-3.1-24b-instruct:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating AI summary: {str(e)}"

def summarize_youtube_video(youtube_url, num_sentences=5):
    """Main function to summarize a YouTube video's transcription."""
    try:
        video_id = extract_video_id(youtube_url)
        transcript = get_transcript(video_id)
        
        # Get video title for context
        try:
            yt = pytube.YouTube(youtube_url)
            video_title = yt.title
            
        except Exception as e:
            video_title = "Unknown Title"

        
        # Generate both summaries
        print(Fore.YELLOW + f"Generating AI summary with {num_sentences} sentences...")
        ai_summary = summarize_text_ai(transcript, video_title, num_sentences)
        
        print(Fore.YELLOW + f"Generating NLTK summary with {num_sentences} sentences...")
        nltk_summary = summarize_text_nltk(transcript, num_sentences)
        
        return {
            "video_title": video_title,
            "video_id": video_id,
            "ai_summary": ai_summary,
            "nltk_summary": nltk_summary,
            "full_transcript_length": len(transcript.split()),
            "nltk_summary_length": len(nltk_summary.split()),
            "ai_summary_length": len(ai_summary.split()) if not ai_summary.startswith("Error") else 0
        }
    except Exception as e:
        return {"error": str(e)}

def format_time(seconds):
    """Convert seconds to a readable time format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def format_number(number):
    """Format large numbers with commas for readability."""
    return "{:,}".format(number)

def print_boxed_text(text, width=80, title=None, color=Fore.WHITE):
    """Print text in a nice box with optional title."""
    wrapper = textwrap.TextWrapper(width=width-4)  # -4 for the box margins
    wrapped_text = wrapper.fill(text)
    lines = wrapped_text.split('\n')
    
    # Print top border with optional title
    if title:
        title_space = width - 4 - len(title)
        left_padding = title_space // 2
        right_padding = title_space - left_padding
        print(color + '┌' + '─' * left_padding + title + '─' * right_padding + '┐')
    else:
        print(color + '┌' + '─' * (width-2) + '┐')
    
    # Print content
    for line in lines:
        padding = width - 2 - len(line)
        print(color + '│ ' + line + ' ' * padding + '│')
    
    # Print bottom border
    print(color + '└' + '─' * (width-2) + '┘')

def print_summary_result(result, width=80):
    """Print the summary result in a nicely formatted way."""
    if "error" in result:
        print_boxed_text(f"Error: {result['error']}", width=width, title="ERROR", color=Fore.RED)
        return
    
    # Terminal width
    terminal_width = width
    
    # Print header with video information
    print("\n" + Fore.CYAN + "=" * terminal_width)
    print(Fore.CYAN + Style.BRIGHT + result['video_title'].center(terminal_width))
    print(Fore.CYAN + "=" * terminal_width + "\n")
    
    # Video metadata section
    print(Fore.YELLOW + Style.BRIGHT + "VIDEO INFORMATION".center(terminal_width))
    print(Fore.YELLOW + "─" * terminal_width)
    
    # Two-column layout for metadata
    col_width = terminal_width // 2 - 2
    
    # Row 3
    print(f"{Fore.GREEN}Video ID: {Fore.WHITE}{result['video_id']:<{col_width}}"
          f"{Fore.GREEN}URL: {Fore.WHITE}https://youtu.be/{result['video_id']}")
    
    print(Fore.YELLOW + "─" * terminal_width + "\n")
    
    # AI Summary section
    ai_compression = "N/A"
    if result['ai_summary_length'] > 0:
        ai_compression = round((1 - result['ai_summary_length'] / result['full_transcript_length']) * 100)
    
    ai_summary_title = f" AI SUMMARY ({result['ai_summary_length']} words, condensed {ai_compression}% from {result['full_transcript_length']} words) "
    
    print(Fore.GREEN + Style.BRIGHT + ai_summary_title.center(terminal_width))
    print(Fore.GREEN + "─" * terminal_width)
    
    # Print the AI summary with proper wrapping
    wrapper = textwrap.TextWrapper(width=terminal_width-4, 
                                  initial_indent='  ', 
                                  subsequent_indent='  ')
    
    # Split AI summary into paragraphs and print each
    ai_paragraphs = result['ai_summary'].split('\n')
    for paragraph in ai_paragraphs:
        if paragraph.strip():  # Skip empty paragraphs
            print(wrapper.fill(paragraph))
            print()  # Empty line between paragraphs
    
    print(Fore.GREEN + "─" * terminal_width + "\n")
    
    # NLTK Summary section
    nltk_compression = round((1 - result['nltk_summary_length'] / result['full_transcript_length']) * 100)
    nltk_summary_title = f" NLTK SUMMARY ({result['nltk_summary_length']} words, condensed {nltk_compression}% from {result['full_transcript_length']} words) "
    
    print(Fore.MAGENTA + Style.BRIGHT + nltk_summary_title.center(terminal_width))
    print(Fore.MAGENTA + "─" * terminal_width)
    
    # Split NLTK summary into paragraphs and wrap each
    paragraphs = result['nltk_summary'].split('. ')
    formatted_paragraphs = []
    
    current_paragraph = ""
    for sentence in paragraphs:
        if not sentence.endswith('.'):
            sentence += '.'
        
        if len(current_paragraph) + len(sentence) + 1 <= 150:  # Arbitrary length for paragraph
            current_paragraph += " " + sentence if current_paragraph else sentence
        else:
            if current_paragraph:
                formatted_paragraphs.append(current_paragraph)
            current_paragraph = sentence
    
    if current_paragraph:
        formatted_paragraphs.append(current_paragraph)
    
    # Print each paragraph
    for paragraph in formatted_paragraphs:
        print(wrapper.fill(paragraph))
        print()  # Empty line between paragraphs
    
    print(Fore.MAGENTA + "─" * terminal_width + "\n")


if __name__ == "__main__":
    # Get terminal width
    try:
        terminal_width = os.get_terminal_size().columns
        # Limit width to reasonable range
        terminal_width = max(80, min(terminal_width, 120))
    except:
        terminal_width = 80  # Default if can't determine
    
    # Print welcome banner
    print(Fore.CYAN + Style.BRIGHT + "\n" + "=" * terminal_width)
    print(Fore.CYAN + Style.BRIGHT + "YOUTUBE VIDEO SUMMARIZER".center(terminal_width))
    print(Fore.CYAN + Style.BRIGHT + "=" * terminal_width + "\n")
    
    youtube_url = input(Fore.GREEN + "Enter YouTube video URL: " + Fore.WHITE)
    
    num_sentences_input = input(Fore.GREEN + "Enter number of sentences for summaries (default 5): " + Fore.WHITE)
    num_sentences = int(num_sentences_input) if num_sentences_input.strip() else 5
    
    print(Fore.YELLOW + "\nFetching and analyzing video transcript... Please wait...\n")
    
    result = summarize_youtube_video(youtube_url, num_sentences)
    print_summary_result(result, width=terminal_width)
