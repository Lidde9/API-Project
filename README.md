# API Project
Analyzing Spotify Music using Spotify and OpenAI APIs

Project Description:

This projects uses the Spotify API to retrieve you discovery weekly playlist and all liked songs from your spotify account, which is an automatically generated playlist updated every monday. The playlist is then parsed through OpenAI API using GPT4 in order to add a genre to each song (which is often missing in Spotify data). The data is thereafter analysed and the following is presented to you in summarised form. The following results is then compared to your full set of liked tracks.

The analysis focuses on the following (among other):
- Popularity of the songs (by analysing temperature data)
- How mainstream the music is (by analysing follower data)
- Most common genres (using Spotify data and data enrichment by OpenAI API)

The Project includes the following:

1) Installation/import of libraries
2) Spotify API to retrieve data         ---------- [ADD YOUR SPOTIFY DETAILS HERE]
3) OpenAI API to add genre to playlist   ------[ADD YOUR OPENAI KEY HERE]
4) Data Cleaning
5) Data Analysis
6) OpenAI API to add story to playlist
7) OpenAI API to summarize share of electronic music