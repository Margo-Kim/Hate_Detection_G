import requests
import json
import pandas as pd

api_key = 'AIzaSyBNs2rWzil8Q0FDiTAW6_IopgQmhh7TLBc'
video_id = 'hQx3I__MFJs&t=3s'

url = f'https://www.googleapis.com/youtube/v3/commentThreads?part=snippet,id&videoId={video_id}&key={api_key}&order=relevance&maxResults=100'
response = requests.get(url).json()

# comment1 = response['items'][0]['snippet']['textDisplay']
# comment = response.get('nextPageToken')[0]
# like_count = response['items'][0]['snippet']['topLevelComment']['snippet']['likeCount']

# print(comment)
all_comments = []
next_page_token = None

while True:
    if next_page_token:
        request_url = f'{url}&pageToken={next_page_token}'
    else:
        request_url = url
        
    response = requests.get(request_url).json()
    
    all_comments.extend(response['items'])
    next_page_token = response['nextPageToken']
    
    if not next_page_token or len(all_comments) >= 1000:  
        break
    

sort_comments = sorted(all_comments, key=lambda x : x['snippet']['topLevelComment']['snippet']['likeCount'], reverse=True)
top_100 = sort_comments[:100]
df = pd.DataFrame(columns = ['comment', 'likes'])

for comment in top_100:
    text = comment['snippet']['topLevelComment']['snippet']['textDisplay']
    likes = comment['snippet']['topLevelComment']['snippet']['likeCount']
    
    df = df.append({'comment' : text,
                    'likes' : likes
        
    }, ignore_index=True
        
    )
    
print(df['comment'][0:5])
    
# save = df.to_csv('youtube_comment.csv', encoding = 'utf-8')
