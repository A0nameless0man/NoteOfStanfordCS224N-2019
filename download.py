import requests
filelist = open("./FileNeeded",'r')
prefix = "./download/"
for url in filelist:
    rep = requests.get(url)
    length = (rep.headers.get('content-length'))
    now = 0
    f = open(prefix+url.split('/')[-1],'wb')
    for ch in rep.iter_content(chunk_size = 2391975):
        if ch:
            now+=len(ch)
            print("Total:" + str(length)+" Cur: "+str(now),end='\r')
            f.write(ch)
