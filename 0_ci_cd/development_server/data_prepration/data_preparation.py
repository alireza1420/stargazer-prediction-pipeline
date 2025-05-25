import requests
import pandas as pd
import time
import json
import re


# Configuration
GITHUB_TOKEN = 'github_pat_11AKHNO6Y0H0tVpqyuywLQ_el28gnuG0nEJhibjauwxrnxuI99ENbpEMmONR--' # Paste your token
HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'}
MAX_REPOS = 1000  # We want top 1000 repos
MIN_STARS = 50    # Minimum 50 stars requirement

def get_top_repositories():
    repos = []
    page = 1
    per_page = 100  # Max allowed by GitHub API
    
    while len(repos) < MAX_REPOS:
        url = f'https://api.github.com/search/repositories?q=stars:>={MIN_STARS}&sort=stars&order=desc&page={page}&per_page={per_page}'
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break
            
        data = response.json()
        if not data['items']:
            break
            
        repos.extend(data['items'])
        print(f"Fetched {len(repos)}/{MAX_REPOS} repositories")
        
        # Avoid hitting rate limit (30 requests/minute for authenticated users)
        time.sleep(1)
        page += 1
    
    return repos[:MAX_REPOS]

# Fetch and save data
top_repos = get_top_repositories()
with open('github_top_repos_raw.json', 'w', encoding='utf-8') as f:
    json.dump(top_repos, f, ensure_ascii=False, indent=4)


# load the row data
with open('github_top_repos_raw.json', 'r', encoding='utf-8') as f:
    top_repos_loaded = json.load(f)

print(type(top_repos_loaded))  # <class 'list'>
print(type(top_repos_loaded[0]))  # <class 'dict'>


def get_count_with_pagination(url):
    try:
        first_page_url = f"{url}?per_page=1&page=1"
        response = requests.get(first_page_url, headers=HEADERS)

        if response.status_code != 200:
            return 0

        # Parse Link header to get the last page number
        link_header = response.headers.get('Link', '')
        match = re.search(r'page=(\d+)>; rel="last"', link_header)

        if match:
            return int(match.group(1))
        else:
            # If no pagination, return the count of items in the response
            return len(response.json())  # 0 或 1
    except:
        return 0

def get_contributor_count(url):
    url = f"{url}?anon=true&per_page=1&page=1002"  # Use a very large page number to trick GitHub into returning to the Link header

    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            return 0

        # Parse Link header to get the last page number
        link_header = response.headers.get('Link', '')
        match = re.search(r'page=(\d+)>; rel="last"', link_header)
        if match:
            last_page = int(match.group(1))
            return last_page 
        else:
            # If no pagination, return the count of items in the response
            data = response.json()
            return len(data) if isinstance(data, list) else 0

    except Exception as e:
        return 0

def get_readme_size(repo_full_name):
    try:
        url = f'https://api.github.com/repos/{repo_full_name}/readme'
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return response.json().get('size', 0)
    except:
        pass
    return 0

def extract_features(repo_data):
    features = {
        'name': repo_data['name'],
        'full_name': repo_data['full_name'],
        'stars': repo_data['stargazers_count'],
        'forks': repo_data['forks_count'],
        'watchers': repo_data['watchers_count'],
        'open_issues': repo_data['open_issues_count'],
        'size': repo_data['size'],
        'has_wiki': int(repo_data['has_wiki']),
        'has_projects': int(repo_data['has_projects']),
        'has_downloads': int(repo_data['has_downloads']),
        'is_fork': int(repo_data['fork']),
        'archived': int(repo_data['archived']),
        'language': repo_data['language'],
        'license': repo_data['license']['key'] if repo_data['license'] else None,
        'created_at': repo_data['created_at'],
        'updated_at': repo_data['updated_at'],
        'pushed_at': repo_data['pushed_at'],

        # New features
        'has_description': int(bool(repo_data['description'])),
        'has_homepage': int(bool(repo_data['homepage'])),
        'topic_count': len(repo_data['topics']) if 'topics' in repo_data and isinstance(repo_data['topics'], list) else 0,
        'has_discussions': int(repo_data.get('has_discussions', False)),
        'is_template': int(repo_data.get('is_template', False)),
        'allow_forking': int(repo_data.get('allow_forking', True)),
        'visibility': repo_data.get('visibility', 'public'),

        # Counts with pagination
        'subscribers_count': get_count_with_pagination(repo_data['subscribers_url']),
        'contributors_count': get_contributor_count(repo_data['contributors_url']),
        'commits_count': get_count_with_pagination(repo_data['commits_url'].split('{')[0]),
        'readme_size': get_readme_size(repo_data['full_name'])
    }
    return features

# Process all repositories
c=1
features_list = []
for repo in top_repos_loaded:
    features = extract_features(repo)
    features_list.append(features)
    print(c)
    c+=1
    print(features)

features_df = pd.DataFrame(features_list)
features_df.to_csv('github_repo_features_new.csv', index=False)