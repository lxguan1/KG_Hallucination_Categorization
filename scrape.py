import requests

API_KEY = "gMFtjzdewnFbupHUG8ACHhN5N2mIymaz0KU8b6yP"
BASE_URL = "https://api.govinfo.gov"

# Directly embed date in the URL instead of using query params
DATE = "2023-01-01T00:00:00Z"  # Adjust as needed
search_url = f"{BASE_URL}/collections/CHRG/{DATE}?offset=0&pageSize=100&api_key={API_KEY}"

response = requests.get(search_url)

if response.status_code == 200:
    data = response.json()
    for hearing in data.get("packages", []):
        package_id = hearing["packageId"]
        title = hearing.get("title", "No Title")
        date = hearing.get("dateIssued", "No Date")

        print(f"ID: {package_id}")
        print(f"Title: {title}")
        print(f"Date: {date}")
        print("-----")

        # Fetch metadata for each hearing
        package_url = f"{BASE_URL}/packages/{package_id}?api_key={API_KEY}"
        package_info = requests.get(package_url).json()

        # Get the .txt file link
        text_url = package_info.get("download", {}).get("txtLink")

        print(f"TXT Link for {package_id}: {text_url}")  # Debugging

        if text_url:
            txt_data = requests.get(text_url).text
            filename = f"{package_id}.txt"

            # Save the file
            with open(filename, "w", encoding="utf-8") as f:
                f.write(txt_data)

            print(f"Downloaded: {filename}")
        else:
            print(f"❌ No TXT file available for {package_id}")

else:
    print("Error:", response.status_code, response.text)
