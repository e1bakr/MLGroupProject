import requests
import json


KEY = "AIzaSyCNUjViUGMC9OQnuq7YiI1DppOY7_X2S8c"
SEARCH_CALL = "https://youtube.googleapis.com/youtube/v3/search?part=snippet&type=video&maxResults=300"


def get_videos(search_term):
    query = "&q=" + search_term
    key = "&key=" + KEY
    
    videos = []
    
    print(SEARCH_CALL + query + key)
    results = requests.get(SEARCH_CALL + query + key).json()
    
    for i in results["items"]:
        videos.append(i)
    
    nextPage = "&pageToken=" + results["nextPageToken"]
    
    for i in range(0, 5):
        print("call : " + str(i))
        print(SEARCH_CALL + query + key + nextPage)
        results = requests.get(SEARCH_CALL + query + key + nextPage).json()
        nextPage = "&pageToken=" + results["nextPageToken"]
        for i in results["items"]:
            videos.append(i)
    
        
    print(len(videos))
    counter = 0
    for i in videos:
            #print(j["id"]["videoId"])
            counter += 1
    # string = json.dumps(videos)
    # file = open("videos.json", "w")
    # file.write(string)
    # file.close()
    return videos
    

def load_topics():
    topicsArray = []
    # file = open("videos.json", "r")
    # data = json.load(file)
    # topicsArray.append(["zip",data])

    topicsArray.append(["food",get_videos("food")])
    topicsArray.append(["football match",get_videos("football match")])
    topicsArray.append(["ocean documentary",get_videos("ocean documentary")])
    topicsArray.append(["make up tutorial",get_videos("make up tutorial")])
    topicsArray.append(["melting ice caps",get_videos("melting ice caps")])


    data = []
    for topic in topicsArray:
        for video in topic[1]:
            id = video["id"]["videoId"]
            img = video["snippet"]["thumbnails"]["default"]
            top = topic[0]
            data.append({
                "video": id,
                "image": img,
                "topic" : top
            })

    file = open("data_image_urls.json", "w")
    file.write(json.dumps(data))
    file.close()
    


if __name__ == "__main__":
    load_topics()