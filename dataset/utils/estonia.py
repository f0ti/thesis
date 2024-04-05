import requests
from bs4 import BeautifulSoup
import re

def parse_response(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    options = {}
    items = soup.find_all('li')
    for index, item in enumerate(items):
        link = item.find('a')
        if link:
            file_name = link.text.strip()
            file_url = link['href']
            if (file_size := re.search(r'\((.*?)\)', item.text)) is not None:
                file_size = file_size.group(1)
            else:
                file_size = "unknown size"
            options[index] = {
                'filename': f"{file_name}",
                "size": file_size,
                'url': file_url
            }
    return options

def print_options(options):
    for key, value in options.items():
        print(f"{key}. {value['filename']} ({value['size']})")

def choose_option(options):
    while True:
        print_options(options)
        choice = input("Enter the number of the file you want to download (or 'q' to quit): ")
        if choice.isdigit():
            choice = int(choice)
            if choice in options:
                return options[choice]
        elif choice.lower() == 'q':
            return None
        print("Invalid input. Please enter a valid number or 'q' to quit.")

class ShowRequest:
    def __init__(self, x, y):
        self.base_show_url = "https://geoportaal.maaamet.ee/index.php?lang_id=2&plugin_act=otsing&andmetyyp=lidar_laz_madal&dl=1&no_cache=660fbc1f272f7&page_id=664"
        self.x = x
        self.y = y

    def get(self):
        response = requests.get(self.base_show_url, params={'kaardiruut': f"{self.x}{self.y}"})
        # check if "File not found" is in the response
        if "File not found" in response.text:
            print("File not found:", self.x, self.y)
            return
        
        if response.status_code != 200:
            print("Failed to get the page")
            return
        else:
            return response.text

class DownloadRequest:
    def __init__(self, choice):
        self.base_download_url = "https://geoportaal.maaamet.ee/"
        self.url = choice['url']
        self.file_name = choice['filename']

    def get(self):
        print("Requesting", self.file_name)
        response = requests.get(f"{self.base_download_url}{self.url}")

        print("Saving", self.file_name)
        print("URL:", response.url)
        if response.status_code != 200:
            print("Failed to download", self.file_name)
            return
        else:
            with open(self.file_name, mode='wb') as localfile:
                localfile.write(response.content)


# the filename is {X}{Y}_2019_madal.laz
X_min = 470
X_max = 584
Y_min = 375
Y_max = 619  # does not include an island in the north
X = range(X_min, X_max+1)
Y = range(Y_min, Y_max+1)

def do_one(x, y):
    sr = ShowRequest(x, y)
    show_content = sr.get()
    if not show_content:
        return
    options = parse_response(show_content)
    selected_option = choose_option(options)
    if not selected_option:
        print("No choice, skipping", x, y)
        return
    else:
        dr = DownloadRequest(selected_option)
        dr.get()

def do_all():
    for x, y in zip(X, Y):
        do_one(x, y)

do_all()
