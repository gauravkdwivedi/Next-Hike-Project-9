import pickle

import streamlit_authenticator as stauth

# User authentication setup
names = ['Gaurav Kumar']
usernames = ['gauravkumar']
passwords = ['xxxx']

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = 'hashed_pw.pkl'
with open(file_path, 'wb') as f:
    pickle.dump(hashed_passwords, f)