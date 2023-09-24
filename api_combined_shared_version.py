#!/usr/bin/env python
# coding: utf-8

# # API Project - Using Spotify and ChatGPT API

# Project Description:
# 
# This projects uses the Spotify API to retrieve you discovery weekly playlist and all liked songs from your spotify account, which is an automatically generated playlist updated every monday. The playlist is then parsed through OpenAI API using GPT4 in order to add a genre to each song (which is often missing in Spotify data). The data is thereafter analysed and the following is presented to you in summarised form. The following results is then compared to your full set of liked tracks.
# 
# - Popularity of the songs (by analysing temperature data)
# - How mainstream the music is (by analysing follower data)
# - Most common genres (using Spotify data and data enrichment by OpenAI API)

# The Project includes the following:
# 
# 1) Installation/import of libraries
# 2) Spotify API to retrieve data         ---------- [ADD YOUR SPOTIFY DETAILS HERE]
# 3) OpenAI API to add genre to playlist   ------[ADD YOUR OPENAI KEY HERE]
# 4) Data Cleaning
# 5) Data Analysis
# 6) OpenAI API to add story to playlist
# 7) OpenAI API to summarize share of electronic music

# # 1) Installation and Libraries

# In[8]:


# Install OpenAI Python Library


# In[1]:


pip install openai


# In[2]:


# Install Spotify Python Library


# In[3]:


pip install spotipy


# In[1]:


# Import Open-AI API
import openai

# Import Spotify API
import spotipy  
from spotipy.oauth2 import SpotifyOAuth  
from spotipy.oauth2 import SpotifyClientCredentials

# Data Analysis and General Libraries
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # 2) Spotify API 

# In[2]:


url = "https://api.spotify.com"


# ### Retrieve Discover Weekly 30 Songs

# In[3]:


# Initialize Spotipy with OAuth
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='[UPDATE CLIENT ID]',
                                               client_secret='[CLIENT SECRET]',
                                               redirect_uri='http://localhost:8080/',
                                               scope='user-library-read user-read-private'))

# Get the "Discover Weekly" playlist ID
playlist = sp.user_playlist('eldrone[USERNAME HERE]', '[UPDATE YOUR PLAYLIST HERE]https://open.spotify.com/playlist/37i9dQZEVXcDMxxJ5B9T5B?si=82a31e5112fe41b6')

# Get the tracks in the "Discover Weekly" playlist
tracks = playlist['tracks']['items']

# Create empty track info list
track_info_list = []

# Loop through the tracks and retrieve all the info
for idx, item in enumerate(tracks[:30], start=1):
    track = item['track']
    track_name = track['name']
    artists = ", ".join([artist['name'] for artist in track['artists']])
    
    # Retrieve track details including artist ID
    track_details = sp.track(track['id'])
    
    # Get artist ID from the first artist of the track
    artist_id = track_details['artists'][0]['id']
    
    # Retrieve artist details
    artist_details = sp.artist(artist_id)
    
    # Get artist genres, popularity, and followers
    genres = artist_details.get('genres', ['Genre not available'])
    popularity = artist_details.get('popularity', 'Popularity not available')
    followers = artist_details.get('followers', {}).get('total', 'Followers not available')
    
    track_uris = [track['uri']]
    audio_features = sp.audio_features(track_uris)
    
    # Extract audio features from the response
    if audio_features:
        acousticness = audio_features[0].get('acousticness')
        danceability = audio_features[0].get('danceability')
        tempo = audio_features[0].get('tempo')
        instrumentalness = audio_features[0].get('instrumentalness')
        liveness = audio_features[0].get('liveness')
        loudness = audio_features[0].get('loudness')
        speechiness = audio_features[0].get('speechiness')
        time_signature = audio_features[0].get('time_signature')
        valence = audio_features[0].get('valence')
    else:
        # Handle the case where audio features are not available
        acousticness = 'N/A'
        danceability = 'N/A'
        tempo = 'N/A'
        instrumentalness = 'N/A'
        liveness = 'N/A'
        loudness = 'N/A'
        speechiness = 'N/A'
        time_signature = 'N/A'
        valence = 'N/A'
    
    print(f"{idx}. Song: {track_name}\n   Artist(s): {artists}\n   Genre(s): {', '.join(genres)}\n Popularity: {popularity}\n Followers: {followers}")
    
    track_info = {
        "Song": track_name,
        "Artist(s)": artists,
        "Genre(s)": genres,
        "Popularity": popularity,
        "Followers": followers,
        "Acousticness": acousticness,
        "Danceability": danceability,
        "Tempo": tempo,
        "Instrumentalness": instrumentalness,
        "Liveness": liveness,
        "Loudness": loudness,
        "Speechiness": speechiness,
        "Time Signature": time_signature,
        "Valence": valence
    }
    
    # Append the track information to the list
    track_info_list.append(track_info)

print(track_info_list)


# In[4]:


dw_data = pd.DataFrame(track_info_list)


# In[6]:


# Take only the song, artist and genre to send through OpenAI API
dw_data_gpt = dw_data[["Song","Artist(s)","Genre(s)"]]


# In[1]:


df_data_gpt


# ### Retrieve all Liked Songs

# In[8]:


"""

from requests.exceptions import ReadTimeout

# Initialize Spotipy with OAuth
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='[UPDATE THIS]',
                                               client_secret='[UPDATE THIS]',
                                               redirect_uri='http://localhost:8080/',
                                               scope='user-library-read user-read-private'))

# Initialize variables
track_info_list = []
offset = 0
limit = 50  # Adjust this value based on your needs
max_retries = 3  # Maximum number of retry attempts

# Function to get liked songs with retries
def get_liked_songs_with_retry():
    retries = 0
    while retries < max_retries:
        try:
            # Get the next batch of liked songs
            playlist = sp.current_user_saved_tracks(limit=limit, offset=offset)
            liked_songs = playlist['items']
            return liked_songs
        except ReadTimeout as e:
            print(f"Request timed out. Retrying ({retries + 1}/{max_retries})...")
            retries += 1

    return []

# Loop through the liked songs using pagination
while True:
    # Get the next batch of liked songs with retries
    liked_songs = get_liked_songs_with_retry()

    if not liked_songs:
        break  # No more songs to retrieve

    # Process each song in the batch
    for idx, item in enumerate(liked_songs, start=offset + 1):
        track = item['track']
        track_name = track['name']
        artists = ", ".join([artist['name'] for artist in track['artists']])

        # Retrieve track details including artist ID
        track_details = sp.track(track['id'])

        # Get artist ID from the first artist of the track
        artist_id = track_details['artists'][0]['id']

        # Retrieve artist details
        artist_details = sp.artist(artist_id)

        # Get artist genres, popularity, and followers with .get
        genres = artist_details.get('genres', ['Genre not available'])
        popularity = artist_details.get('popularity', 'Popularity not available')
        followers = artist_details.get('followers', {}).get('total', 'Followers not available')

        track_uris = [track['uri']]
        audio_features = sp.audio_features(track_uris)

        # Check if audio features are available for the song with .get
        if audio_features and audio_features[0]:
            acousticness = audio_features[0].get('acousticness', 'N/A')
            danceability = audio_features[0].get('danceability', 'N/A')
            tempo = audio_features[0].get('tempo', 'N/A')
            instrumentalness = audio_features[0].get('instrumentalness', 'N/A')
            liveness = audio_features[0].get('liveness', 'N/A')
            loudness = audio_features[0].get('loudness', 'N/A')
            speechiness = audio_features[0].get('speechiness', 'N/A')
            time_signature = audio_features[0].get('time_signature', 'N/A')
            valence = audio_features[0].get('valence', 'N/A')
        else:
            # Handle the case where audio features are not available
            acousticness = 'N/A'
            danceability = 'N/A'
            tempo = 'N/A'
            instrumentalness = 'N/A'
            liveness = 'N/A'
            loudness = 'N/A'
            speechiness = 'N/A'
            time_signature = 'N/A'
            valence = 'N/A'

        print(f"{idx}. Song: {track_name}\n   Artist(s): {artists}\n   Genre(s): {', '.join(genres)}\n Popularity: {popularity}\n Followers: {followers}")

        track_info = {
            "Song": track_name,
            "Artist(s)": artists,
            "Genre(s)": genres,
            "Popularity": popularity,
            "Followers": followers,
            "Acousticness": acousticness,
            "Danceability": danceability,
            "Tempo": tempo,
            "Instrumentalness": instrumentalness,
            "Liveness": liveness,
            "Loudness": loudness,
            "Speechiness": speechiness,
            "Time Signature": time_signature,
            "Valence": valence
        }

        # Append the track information to the list
        track_info_list.append(track_info)

    offset += limit

print(track_info_list)

"""


# ## Save the full liked list after importing

# """
# 
# file_path = "track_info_list.json"
# 
# # Save the list of dictionaries as a JSON file
# with open(file_path, 'w', encoding='utf-8') as json_file:
#     json.dump(track_info_list, json_file, ensure_ascii=False, indent=4)
# 
# print(f"Data saved as {file_path}")
# 
# """

# ## Import all liked songs saved

# In[9]:


# Specify the path to your JSON file
json_file_path = 'track_info_list.json'  # Replace 'your_file.json' with the actual file path

# Open and load the JSON file
with open(json_file_path, 'r') as json_file:
    data_all_list = json.load(json_file)


# In[10]:


dw_data_all_liked = pd.DataFrame(data_all_list)


# # 3) OpenAI API

# In[12]:


# Set the API Key
openai.api_key = '[NEED TO PAY FOR OPENAI API TO GET HIS]'


# ## List of Models Available
# List of Available Models (ranked from cheapest to most expensive)
# 
# - "gpt-3.5-turbo"
# - "text-davinci-003"
# - "gpt-4"

# In[13]:


# Send the question to chat GPT in order to get a reponse back
response = openai.ChatCompletion.create(
    # What model to use
    model="gpt-4",
    # The full message including the role of the chat assistant and the first message
    messages=[
        {"role": "system", "content": "You are specialist in music and categorizing genres of songs"},
        {"role": "user", "content": f"Can you add the genre of each of these songs and return the updated list. Please only change the genre for those with no genre. Answer with only the updated list in json format. {dw_data_gpt}"}
    ],
    # Max token to set price caps
    max_tokens=3000,
    # Temperature set, indicates how random the model should be in picking a word (0 is not randomness and 1 is wild)
    temperature=0.1
)


# In[14]:


# Extract only the message response
dw_data_gadd = response["choices"][0]["message"]["content"]


# In[15]:


# Create a json file from the message (which comes as a string in JSON format)
dw_data_gadd_json = json.loads(dw_data_gadd)


# In[16]:


# Create a DataFrame of the JSON file
dw_data_gadd_df = pd.DataFrame(dw_data_gadd_json)


# In[17]:


# Overwrite the first three columns received from OpenAI to the full DataFrame from Spotify
dw_data["Genre(s)+GPT"] = dw_data_gadd_df[["Genre(s)"]]

# Put the new Genre in 4th position in data

desired_position = 3  # 0-based index, so 3 is the 4th position

# Get the list of column names
columns = dw_data.columns.tolist()

# Remove the column you want to move from the list of columns
column_to_move = columns.pop(-1)  # Remove the last column

# Insert the column at the desired position
columns.insert(desired_position, column_to_move)

# Reorder the DataFrame columns
dw_data = dw_data[columns]


# # 4) Data Cleaning

# In[40]:


# Assuming dw_data_all_liked is your DataFrame
columns_to_convert = ["Acousticness", "Danceability", "Tempo", "Instrumentalness", "Liveness", "Loudness", "Speechiness", "Time Signature", "Valence"]

# Loop through each column and perform the operations
for column in columns_to_convert:
    # Count the 'N/A' values before replacement
    n_na_values = (dw_data_all_liked[column] == 'N/A').sum()
    print(f"Number of 'N/A' values in the '{column}' column before replacement: {n_na_values}")
    
    # Replace 'N/A' with NaN
    dw_data_all_liked[column].replace('N/A', np.nan, inplace=True)
    
    # Convert the column to float
    dw_data_all_liked[column] = dw_data_all_liked[column].astype(float)
    
    # Count the 'NaN' values after replacement
    n_nan_values = dw_data_all_liked[column].isna().sum()
    print(f"Number of 'NaN' values in the '{column}' column after replacement: {n_nan_values}")


# # 5) Data Analysis

# In[41]:


dw_data_all_liked["Valence"].mean()


# In[42]:


dw_data_all_liked["Genre(s)"].value_counts().head(60)


# In[43]:


# Calculate the mean values
mean_values = {
    "Followers_mean": dw_data_all_liked["Followers"].mean(),
    "Popularity_mean": dw_data_all_liked["Popularity"].mean(),
    "Acousticness_mean": dw_data_all_liked["Acousticness"].mean(),
    "Danceability_mean": dw_data_all_liked["Danceability"].mean(),
    "Tempo_mean": dw_data_all_liked["Tempo"].mean(),
    "Instrumentalness_mean": dw_data_all_liked["Instrumentalness"].mean(),
    "Liveness_mean": dw_data_all_liked["Liveness"].mean(),
    "Loudness_mean": dw_data_all_liked["Loudness"].mean(),
    "Speechiness_mean": dw_data_all_liked["Speechiness"].mean(),
    "Valence_mean": dw_data_all_liked["Valence"].mean()
}

# Create a new DataFrame
mean_df = pd.DataFrame(mean_values, index=[0])

pd.options.display.float_format = '{:.6f}'.format

mean_df = mean_df.T

mean_df.columns = ["All Liked 2500 Songs"]


# In[44]:


mean_values_dw = {
    "Followers_mean": dw_data["Followers"].mean(),
    "Popularity_mean": dw_data["Popularity"].mean(),
    "Acousticness_mean": dw_data["Acousticness"].mean(),
    "Danceability_mean": dw_data["Danceability"].mean(),
    "Tempo_mean": dw_data["Tempo"].mean(),
    "Instrumentalness_mean": dw_data["Instrumentalness"].mean(),
    "Liveness_mean": dw_data["Liveness"].mean(),
    "Loudness_mean": dw_data["Loudness"].mean(),
    "Speechiness_mean": dw_data["Speechiness"].mean(),
    "Valence_mean": dw_data["Valence"].mean()
}

# Create a new DataFrame
mean_df_dw = pd.DataFrame(mean_values_dw, index=[0])

mean_df_dw = mean_df_dw.T

mean_df_dw.columns = ["Discover Weekly 30"]


# In[45]:


combined_df = pd.concat([mean_df, mean_df_dw], axis=1)


# In[46]:


combined_df


# In[47]:


combined_df["Difference"] = combined_df["Discover Weekly 30"] - combined_df["All Liked 2500 Songs"]


# In[48]:


combined_df


# In[49]:


combined_df["Difference Pct"] = (combined_df["Discover Weekly 30"] - combined_df["All Liked 2500 Songs"]) / combined_df["All Liked 2500 Songs"]


# In[50]:


combined_df


# In[51]:


combined_df_json = combined_df.to_json(orient='records')


# In[52]:


combined_df_json


# In[81]:


plt.style.use("ggplot")


# In[86]:


# Colors in graph
colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

# Create a bar plot
plt.figure(figsize=(8, 8))
#plt.title('Percentage Difference between All Liked 2500 Songs and Discover Weekly 30', fontsize=14)
combined_df['Difference Pct'].plot(kind='bar', color=colors, alpha=0.7)
plt.xlabel('Category', fontsize=16)
plt.ylabel('Percentage Difference', fontsize=16)
plt.xticks(rotation=60, ha='right', fontsize=13)
plt.tight_layout()
plt.show()


# In[69]:


# Colors in graph
colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

# Create a bar plot
plt.figure(figsize=(12, 8))  # Increase the figure size if necessary
plt.title('Percentage Difference between All Liked 2500 Songs and Discover Weekly 30', fontsize=16)  # Increase title font size
combined_df['Difference Pct'].plot(kind='bar', color=colors, alpha=0.7)
plt.xlabel('Category', fontsize=14)  # Increase x-axis label font size
plt.ylabel('Percentage Difference', fontsize=14)  # Increase y-axis label font size
plt.xticks(rotation=60, ha='right', fontsize=12)  # Increase x-axis tick label font size
plt.yticks(fontsize=12)  # Increase y-axis tick label font size
plt.tight_layout()
plt.show()


# In[56]:


dw_data.info()


# In[57]:


dw_data["Genre(s)+GPT"].value_counts()


# In[58]:


plt.style.available


# In[59]:


# This is matplotlib syntax
# Figure and axis to keep track of in 

# Streamlit 
# Set the color palette

sns.set_palette("mako")

plt.style.use("seaborn-v0_8-dark")

fig, ax = plt.subplots(3,3, figsize=(12,8))
plt.suptitle("10 Years of My Music - Audio Features ", fontsize = 34)
ax[0][0].title.set_text("Popularity Index")
sns.histplot(x="Popularity", data=dw_data_all_liked, ax=ax[0][0]) # Row and column position in ax
ax[0][1].title.set_text("Artists Followers")
sns.boxplot(x="Followers", data=dw_data_all_liked, ax=ax[0][1])
ax[0][2].title.set_text("Acoustic Index")
sns.histplot(x="Acousticness", data=dw_data_all_liked, ax=ax[0][2])
ax[1][0].title.set_text("Wanna Dance Index")
sns.histplot(x="Danceability", data=dw_data_all_liked, ax=ax[1][0])
ax[1][1].title.set_text("Tempo BPM")
sns.histplot(x="Tempo", data=dw_data_all_liked, ax=ax[1][1])
ax[1][2].title.set_text("Instruments Index")
sns.histplot(x="Instrumentalness", data=dw_data_all_liked, ax=ax[1][2])
ax[2][0].title.set_text("Lively Index")
sns.histplot(x="Liveness", data=dw_data_all_liked, ax=ax[2][0])
ax[2][1].title.set_text("Speech Index")
sns.histplot(x="Speechiness", data=dw_data_all_liked, ax=ax[2][1])
ax[2][2].title.set_text("Happy or Sad Index")
sns.histplot(x="Valence", data=dw_data_all_liked, ax=ax[2][2])

plt.tight_layout()
plt.show()


# In[335]:


# Set the color palette
sns.color_palette("mako", as_cmap=True)

# Set the style
plt.style.use("seaborn-v0_8-dark")

# Create the subplots
fig, ax = plt.subplots(3, 3, figsize=(14, 9))
plt.suptitle("10 Years of My Music - Audio Features", fontsize=34)

# Define colors from the palette
colors = sns.color_palette("mako", n_colors=9)  # Choose 9 colors for 9 plots

# Plot each feature with different colors
ax[0][0].title.set_text("Popularity Index")
sns.histplot(x="Popularity", data=dw_data_all_liked, ax=ax[0][0], color=colors[0])
ax[0][1].title.set_text("Artists Followers")
sns.boxplot(x="Followers", data=dw_data_all_liked, ax=ax[0][1], color=colors[1])
ax[0][2].title.set_text("Acoustic Index")
sns.histplot(x="Acousticness", data=dw_data_all_liked, ax=ax[0][2], color=colors[2])
ax[1][0].title.set_text("Wanna Dance Index")
sns.histplot(x="Danceability", data=dw_data_all_liked, ax=ax[1][0], color=colors[3])
ax[1][1].title.set_text("Tempo BPM")
sns.histplot(x="Tempo", data=dw_data_all_liked, ax=ax[1][1], color=colors[4])
ax[1][2].title.set_text("Instruments Index")
sns.histplot(x="Instrumentalness", data=dw_data_all_liked, ax=ax[1][2], color=colors[5])
ax[2][0].title.set_text("Lively Index")
sns.histplot(x="Liveness", data=dw_data_all_liked, ax=ax[2][0], color=colors[6])
ax[2][1].title.set_text("Speech Index")
sns.histplot(x="Speechiness", data=dw_data_all_liked, ax=ax[2][1], color=colors[7])
ax[2][2].title.set_text("Happy or Sad Index")
sns.histplot(x="Valence", data=dw_data_all_liked, ax=ax[2][2], color=colors[8])

# Add a tight layout
plt.tight_layout()

# Show the plot
plt.show()


# In[60]:


dw_data_all_liked.info()


# In[67]:


dw_data_all_liked["Followers"].mean()


# In[68]:


dw_data_all_liked["Followers"].median()


# # 6) Request to summarise the results of the playlist vs. all liked songs

# In[61]:


sound_features = {
    'Followers': "Number of followers for the artist.",
    'Popularity': "The popularity of the artist. The value will be between 0 and 100, with 100 being the most popular. The artist's popularity is calculated from the popularity of all the artist's tracks.",
    'Acousticness': 'A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.',
    'Danceability': 'Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.',
    'Tempo': 'The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.',
    'Instrumentalness': 'Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.',
    'Liveness': 'Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.',
    'Loudness': 'The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 dB.',
    'Speechiness': 'Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.',
    'Valence': 'A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).'
}


# In[62]:


combined_df


# In[63]:


dw_genres_gpt = dw_data["Genre(s)+GPT"]


# In[64]:


dw_genres_gpt


# In[298]:


# Send the question to chat GPT in order to get a reponse back
response2 = openai.ChatCompletion.create(
    # What model to use
    model="gpt-4",
    # The full message including the role of the chat assistant and the first message
    messages=[
        {"role": "system", "content": f"You specialize in comparing playlist based on sound features {sound_features}"},
        {"role": "user", "content": f" you are presenting this weeks playlist with these genres {dw_genres_gpt} and sound features {combined_df_json}. The presentation should be three sentences (of which only one mentiones genres) and only include the sound features which are more than 0.1 different and not mention any percentages, numbers or playlists. "}
    ],
    # Max token to set price caps
    max_tokens=3000,
    # Temperature set, indicates how random the model should be in picking a word (0 is not randomness and 1 is wild)
    temperature=1
)


# In[299]:


response2


# In[280]:


response2["choices"][0]["message"]["content"]


# In[65]:


dw_datacopy = dw_data


# In[66]:


dw_datacopy.drop("Genre(s)+GPT", axis=1)


# # 7) Ask for share of electronic tracks

# In[90]:


# Send the question to chat GPT in order to get a reponse back
response3 = openai.ChatCompletion.create(
    # What model to use
    model="gpt-4",
    # The full message including the role of the chat assistant and the first message
    messages=[
        {"role": "system", "content": f"You categorize subgenres into broader genres"},
        {"role": "user", "content": f" Can you summarize of large share of these songs are electronic music or a subgenre of electronic music? Please answer with a number ranging from 0-30, and a short summary of the most popular genres in this weeks playlist {dw_genres_gpt}"}
    ],
    # Max token to set price caps
    max_tokens=3000,
    # Temperature set, indicates how random the model should be in picking a word (0 is not randomness and 1 is wild)
    temperature=1
)


# In[91]:


response3


# In[ ]:




