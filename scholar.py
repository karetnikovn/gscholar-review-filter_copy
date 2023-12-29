import csv

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from bs4 import BeautifulSoup
import pandas as pd
import requests
import re
import logging
import configparser
import time
import openai
import requests
from pdfminer.high_level import extract_text


# Basic configs:
logging.basicConfig(filename='log.log', level=logging.INFO, format='%(asctime)s %(message)s')
config = configparser.ConfigParser()
config.read('config.ini')
openai.api_key = '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'

# Load PICOC terms
picoc = {
    'population': re.compile(config['picoc']['population']) if config['picoc']['population'] != "" else None,
    'intervention': re.compile(config['picoc']['intervention']) if config['picoc']['intervention'] != "" else None,
    'comparison': re.compile(config['picoc']['comparison']) if config['picoc']['comparison'] != "" else None,
    'outcome': re.compile(config['picoc']['outcome']) if config['picoc']['outcome'] != "" else None,
    'context': re.compile(config['picoc']['context']) if config['picoc']['context'] != "" else None
}

# Create a new Chorme session

options = None
if config['default']['binary_location']:
    options = Options()
    options.binary_location = config['default']['binary_location']

driver = webdriver.Chrome(r"C:\Users\karet\Downloads\chromedriver_win32\chromedriver", options=options)
url = "https://scholar.google.com/"
driver.get(url)

# Setting Google Scholar
driver.maximize_window()
time.sleep(1)
driver.find_element_by_id("gs_hdr_mnu").click()
time.sleep(1)
driver.find_element_by_class_name("gs_btnP").click()
time.sleep(1)
driver.find_element_by_id("gs_num-b").click()
time.sleep(1)
driver.find_element_by_css_selector('a[data-v="20"').click()
time.sleep(1)
driver.find_element_by_id("gs_settings_import_some").click()
time.sleep(1)
driver.find_element_by_name("save").click()


def split_content_into_chunks(content, chunk_size=8000):  # chunk_size can be adjusted as needed
    return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import nltk

# Ensure you've downloaded the Punkt tokenizer models
nltk.download('punkt')

def clean_html_content(html_content):
    # Use BeautifulSoup to parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract all text from the soup object
    all_text = soup.get_text(separator=' ', strip=True)

    # Tokenize the text into sentences
    sentences = sent_tokenize(all_text)

    # Initialize an empty list to hold filtered segments
    filtered_segments = []

    # Temporary variable to hold the count of sentences in a segment
    segment_sentence_count = 0

    # Variable to hold text of the current segment
    current_segment = []

    # Iterate over sentences and group them into segments with more than 3 sentences
    for sentence in sentences:
        current_segment.append(sentence)
        segment_sentence_count += 1

        # Check if the current segment has more than 3 sentences
        if segment_sentence_count > 3:
            # Add the segment to the filtered list and reset the count and segment
            filtered_segments.append(' '.join(current_segment))
            segment_sentence_count = 0
            current_segment = []

    # Check if the last segment had more than 3 sentences and add it if it was not added
    if segment_sentence_count > 3:
        filtered_segments.append(' '.join(current_segment))

    # Join the filtered segments back into a single string, separated by double newlines
    clean_text = "\n\n".join(filtered_segments)

    return clean_text


def extract_text_from_pdf(pdf_link):
    try:
        # Download PDF
        response = requests.get(pdf_link)
        response.raise_for_status()  # Check if the request was successful

        file_path = "temp.pdf"
        with open(file_path, "wb") as f:
            f.write(response.content)

        # Extract text from the first two pages
        return extract_text(file_path, page_numbers=[0, 1,2,3])
    except requests.RequestException as e:
        print(f"Error downloading PDF: {e}")
        return ""
    except Exception as e:
        print(f"Error occurred while processing the PDF: {e}")
        return ""


import concurrent.futures
import openai
import time

def analyze_abstract(abstract, max_retries=4, initial_delay=5):
    def request_to_openai(message_content, max_retries, initial_delay):
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = openai.Completion.create(
                    model="gpt-3.5-turbo-instruct",  # Assuming you're using an Instruct model named "text-davinci-003"
                    prompt=message_content,
                    temperature=0,  # Adjust as needed for creativity
                    max_tokens=20
                )
                return response.choices[0].text.strip()
            except openai.error.OpenAIError as e:
                if "The server is overloaded or not ready yet" in str(e) and retry_count < max_retries - 1:
                    wait_time = initial_delay * (2 ** retry_count)  # Exponential backoff
                    print(f"Server error detected. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retry_count += 1
                else:
                    print("Something went wrong in request_to_openai_instruct function")
                    return "NA"
                    # Optionally, y

    # Step 1: Check if the abstract discusses a simulation with LLM models
    print("Asking about methodes...")

    response_1_content = request_to_openai(
        "Is it stated in the paper abstract, that authors developed an agent-based simulation with Large Language Models in order to study social phenomena?"
        "Please give Yes/No as an answer."
        "Abstract starts: " + abstract +
        "Abstract ends.",
        max_retries, initial_delay)

    if not ("yes" in response_1_content.lower()):
        return response_1_content.lower(), None

    # Step 15: Identify the domain of the simulation
    domain = request_to_openai(
        "What is the domain of the simulation described in this abstract (e.g., social, economic, etc.)? Please give max 5 words as an answer."
        "If there is no information to answer this question, state NA. Abstract: " + abstract, max_retries,
        initial_delay)

    """
    # Step 2: Identify the type of simulation
    entities = request_to_openai(
        "Does the abstract mention agents or entities that could represent individuals, groups, or institutions ?"
        "Please give Yes/No/Unclear as an answer."
        "If there is no information to answer this question, state NA. Abstract: " + abstract, max_retries,
        initial_delay)

    print("Asking about interactions...")

    # Step 3: Outlining the main benefits
    interactions = request_to_openai("Does the abstract mention interactions, behaviors,communication, or dynamics that are characteristic of social systems ?"
                                  "Please give Yes/No as an answer."
                                 "If there is no information to answer this question, state NA. Abstract: " + abstract,
                                 max_retries, initial_delay)

    """

    return response_1_content.lower(), domain




import time


def get_abstract_from_chatgpt(content, content_type="html", max_retries=3, initial_delay=5):
    # Preprocess content based on its type
    if content_type == "html":
        cleaned_content = clean_html_content(content)
    elif content_type == "pdf":
        cleaned_content = content  # Assuming PDF content is already text and doesn't need further cleaning
    else:
        raise ValueError(f"Unknown content_type: {content_type}")

    chunks = split_content_into_chunks(cleaned_content)
    for chunk in chunks:
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Construct the role message based on content type
                system_message = f"You are a helpful assistant that extracts abstracts from {content_type.upper()} content."

                response = openai.Completion.create(
                    model="gpt-3.5-turbo-instruct",  # Assuming you're using an Instruct model named "text-davinci-003"
                    prompt=f"Extract abstract from the following {content_type.upper()} content: {chunk}. Output only it and nothing else."
                                    f"If you did not find an abstract, please output 'Not Found'.",
                    temperature=0,  # Adjust as needed for creativity
                    max_tokens=500
                )
                abstract = response.choices[0]['text'].strip()
                if 'not found' not in abstract.lower():
                    return abstract
                break  # Break out of the retry loop if no server error occurred
            except Exception as e:
                if "The server is overloaded or not ready yet" in str(e) and retry_count < max_retries - 1:
                    wait_time = initial_delay * (2 ** retry_count)  # Exponential backoff
                    print(f"Server error detected. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retry_count += 1
                else:
                    raise  # Raise the exception if it's a different error or if retries are exhausted

    # If all chunks were processed and no abstract was found:
    return "Not Found"


# Sometimes a Captcha shows up. It needs to be fixed manually. This function makes the code wait until thisj be fixed
def check_captcha():
    captcha = driver.find_elements_by_css_selector("#captcha")
    captcha += driver.find_elements_by_css_selector("#gs_captcha_f")
    captcha += driver.find_elements_by_css_selector("#g-recaptcha")
    captcha += driver.find_elements_by_css_selector("#recaptcha")
    while captcha:
        logging.info(
            "Captcha found! You need to fill it on browser to continue. Go to the terminal and type 'y' when the Captcha be solved")
        print("Captcha found! You need to fill it on browser to continue...")
        solve = input("Type 'y' when the Captcha be solved: ")
        if solve == "y":
            try:
                driver.find_element_by_id("gs_res_sb_yyc")
                logging.info("Captcha solved, continuing...")
                break
            except:
                print("Captcha not solved, try again! You need to fill it on browser to continue...")
                logging.info("Captcha not solved, try again! You need to fill it on browser to continue...")
        else:
            print("Input error. Try again")


# Filter the PICOC terms inside the Title-Abstract-Keywords

def extract_abstract(content, content_type):
    """Utility function to get abstract and handle errors"""
    try:
        abstract = get_abstract_from_chatgpt(content, content_type=content_type)
        print(abstract)
        return abstract
    except Exception as e:
        print(f"Error occurred: {e}")
        return ''

def is_pdf(link):
    try:
        # Get a small portion of the file (first 5 bytes should suffice for checking the PDF signature)
        response = requests.get(link, stream=True, timeout=5)
        start_of_file = response.content[:5]
        return start_of_file == b'%PDF-'
    except requests.RequestException:
        return False

# Parser HTML
def parser(soup, page, year, number_simulations):
    papers = []
    html = soup.findAll('div', {'class': 'gs_r gs_or gs_scl'})
    for result in html:
        paper = {'Link': result.find('h3', {'class': "gs_rt"}).find('a')['href'], 'Additional link': '', 'Title': '',
                 'Authors': '', 'Abstract': '', 'Cited by': '', 'Cited list': '', 'Related list': '', 'Bibtex': '',
                 'Year': year, 'Google page': page, "Research method?":'', "Simulation Domain":''}


        for a in result.findAll('div', {'class': "gs_fl"})[-1].findAll('a'):
            if a.text != '':
                if a.text.startswith('Cited'):
                    paper['Cited by'] = a.text.rstrip().split()[-1]
                    paper['Cited list'] = url + a['href']

                if a.text.startswith('Related'):
                    paper['Related list'] = url + a['href']
                if a.text.startswith('Import'):
                    paper['Bibtex'] = requests.get(a['href']).text


        if (paper['Cited by'].isdigit() == False):
            continue




        paper['Title'] = result.find('h3', {'class': "gs_rt"}).text
        print(paper['Title'])
        paper['Authors'] = ";".join(
            ["%s:%s" % (a.text, a['href']) for a in result.find('div', {'class': "gs_a"}).findAll('a')])


        if is_pdf(paper["Link"]):
            pdf_text = extract_text_from_pdf(paper["Link"])
            paper["Abstract"] = extract_abstract(pdf_text, content_type="pdf")
        else:
            try:
                driver.get(paper["Link"])
                time.sleep(1)
                papier_page_soup = BeautifulSoup(driver.page_source, 'html.parser')
                html_content = str(papier_page_soup)
                paper["Abstract"] = extract_abstract(html_content, content_type="html")
                driver.back()
                time.sleep(1)
            except:
                paper["Abstract"] = "Not Found"

        # Analyzing the abstract
        is_llm_simulation_abstract, domain= analyze_abstract(paper["Abstract"])



        print(f"Research method? {is_llm_simulation_abstract}")

        # Convert both strings to lower case for case-insensitive comparison
        is_llm_simulation_abstract_lower = is_llm_simulation_abstract.lower()

        # Check for the presence of the words in both strings
        if (("yes" in is_llm_simulation_abstract_lower or "yes." in is_llm_simulation_abstract_lower)):
            number_simulations += 1
            print(f"We have collected {number_simulations} simulations")


        paper["Research method?"] = is_llm_simulation_abstract
        paper["Simulation Domain"] = domain


        try:
            paper["Additional link"] = result.find('div', {'class': "gs_or_ggsm"}).find('a')['href']
        except:
            paper["Additional link"] = ''


        papers.append(paper)


        # Wait 20 seconds until the next request to google
        time.sleep(2)

    return papers, len(html),number_simulations


import time
import logging
from selenium.common.exceptions import NoSuchElementException




if __name__ == '__main__':

    query = config['search']['query']
    year = int(config['search']['start_year'])
    output = config['default']['result_path']

    logging.info("Starting...")
    logging.info("Result path: {}".format(output))
    logging.info("PICOC terms are: {}".format(picoc))
    logging.info("Search query is: {}".format(query))

    with open(output, 'a', newline='') as outcsv:
        csv.writer(outcsv).writerow(['Link', 'Additional link', 'Title', 'Authors', 'Abstract', 'Cited by',
                                     'Cited list', 'Related list', 'Bibtex', 'Year', 'Google page',"Research method?","Simulation Domain"])

    # String search year by year.
    while year <= int(config['search']['end_year']):
        driver.get(url + "scholar?hl=en&q={0}&as_sdt=1&as_vis=1&as_ylo={1}&as_yhi={1}".format(query, year))
        check_captcha()
        page = 1
        total = 0
        number_simulations = 0
        while True:
            print("PARSER IN")

            art, t,number_simulations = parser(BeautifulSoup(driver.page_source, 'lxml'), page, year, number_simulations)

            print("PARSER OUT")
            total += t

            df = pd.DataFrame(art)
            df.to_csv(output, mode='a', header=False, index=False)

            max_retries = 5  # Set the number of retries
            retries = 0

            while retries < max_retries:
                try:
                    # Try to find and click the next page button
                    driver.find_element_by_class_name("gs_ico_nav_next").click()
                    print("ELEMENT WAS FOUND")

                    check_captcha()
                    print("CAPTCHA CHECKED")

                    page += 1
                    print("PAGE UPDATED")

                    break  # If successful, break out of the retry loop
                except NoSuchElementException:
                    # Element not found, wait and then retry
                    print(f"Attempt {retries + 1} failed. Retrying...")
                    retries += 1
                    time.sleep(5)  # Wait before retrying
                except Exception as e:
                    # Handle other exceptions and log the message
                    logging.info(f"An exception occurred: {e}")
                    break  # Break the loop if an unexpected exception occurs

            if retries == max_retries:
                # If the maximum retries have been reached
                logging.info(
                    "No more pages for {} year, total of {} pages and {} articles processed".format(year, page, total))
                year += 1

                print("NO PAGE")
                # Wait 10 seconds until the next page request
                time.sleep(10)


    logging.info("Ending...")
    driver.close()
