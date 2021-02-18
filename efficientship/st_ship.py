import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
import datetime
import pickle
import base64

from math import ceil
from sklearn.metrics import auc

# import ast
# from Crypto.PublicKey import RSA
# from Crypto.Random import get_random_bytes
# from Crypto.Cipher import AES, PKCS1_OAEP


@st.cache
def load_locations(bgn, end, dummy=datetime.datetime.now().minute//5):
    with open('data/location.pkl', 'rb') as f:
        hourly = pickle.load(f)
    hourly = hourly.loc[(hourly['ts'] >= bgn) & (hourly['ts'] <= end)]
    return hourly

@st.cache
def location(hourly, t0, t1):
    """
    """
    hourly = hourly.loc[(hourly['ts'] >= t0) & (hourly['ts'] <= t1)]
    if hourly.empty:
        return
    
    latlon = pdk.Layer(
            "ArcLayer",
            data=hourly,
            get_width=5,
            get_source_position=["lon_origin", "lat_origin"],
            get_target_position=["lon", "lat"],
            get_tilt=10,
            get_source_color='[0,0,255,15]',
            get_target_color='[0,0,255,240]',
            pickable=True,
            auto_highlight=True,
        )

    layers = [latlon]

    r = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v10",
        initial_view_state={"latitude": 1.290, "longitude": 103.85, "zoom": 10, "pitch": 45},
        layers=layers, 
        tooltip={
            'html': '<div>' +
                    '<p>{interval}</p>' +
                    '</div>', 
            'style': {
                'color': 'white'
            }
        },
    )
    return r

# def generate_key_pair():
#     # key
#     key = RSA.generate(2048)
#     private_key = key.export_key()
#     file_out = open("key", "wb")
#     file_out.write(private_key)
#     file_out.close()
#     # public
#     public_key = key.publickey().export_key()
#     file_out = open("key.pub", "wb")
#     file_out.write(public_key)
#     file_out.close()
#     # done
#     return

# def encryptor(data, key):
#     data = data.encode("utf-8")
#     # load public key and random generate session key
#     recipient_key = RSA.import_key(key.read())
#     session_key = get_random_bytes(16)
#     # encrypt the session key with the public RSA key
#     cipher_rsa = PKCS1_OAEP.new(recipient_key)
#     enc_session_key = cipher_rsa.encrypt(session_key)
#     # encrypt the data with the AES session key
#     cipher_aes = AES.new(session_key, AES.MODE_EAX)
#     ciphertext, tag = cipher_aes.encrypt_and_digest(data)
#     # either save/return
#     return enc_session_key, cipher_aes.nonce, tag, ciphertext

# def decryptor(args):
#     enc_session_key, nonce, tag, ciphertext = args
#     # load locally
#     private_key = RSA.import_key(open("key").read())
#     # decrypt the session key with the private RSA key
#     cipher_rsa = PKCS1_OAEP.new(private_key)
#     session_key = cipher_rsa.decrypt(enc_session_key)
#     # decrypt the data with the AES session key
#     cipher_aes = AES.new(session_key, AES.MODE_EAX, nonce)
#     data = cipher_aes.decrypt_and_verify(ciphertext, tag)
#     # done, return message
#     return data.decode("utf-8")

# time frame
# @st.cache()
def get_time_frames(dummy=datetime.datetime.now().minute//5):
    with open('data/segments.pkl', 'rb') as f:
        segments = pickle.load(f)
    segments = segments.sort_values(by='start', ascending=False).reset_index(drop=True)
    bgn = segments['start'].apply(lambda t: pd.Timestamp(t, unit='s', tz='Asia/Singapore'))
    end = segments['end'].apply(lambda t: pd.Timestamp(t, unit='s', tz='Asia/Singapore'))
    ranges = list(map(lambda be: '{} to {}'.format(*be), \
                      zip(pd.Series.dt(bgn).strftime('%d/%m_%H:%M'), pd.Series.dt(end).strftime('%d/%m_%H:%M'))))
    return ranges, bgn.to_numpy(), end.to_numpy()

@st.cache
def load_mfm_data(dummy=datetime.datetime.now().minute//5):
    # load mass flow meter data
    with open('data/mfm.pkl', 'rb') as f:
        mfm_data = pickle.load(f)
    return mfm_data

@st.cache()
def slice(mfm_data, t0, t1):
    mfm_slice = mfm_data[(mfm_data.index >= t0) & (mfm_data.index <= t1)]
    return mfm_slice


if __name__ == '__main__':

    # pub = st.sidebar.file_uploader('select key')
    # username = st.sidebar.text_input('username')
    # password = st.sidebar.text_input('password', type='password')
    # login = st.sidebar.button('login')
    # if login and username != '' and password != '' and pub is not None:
    #     # validate username and password

    #     plain = username
    #     packet = encryptor(plain, pub)
    #     for p in packet:
    #         st.write(p)
    #     decrypted = decryptor(packet)
    #     st.write('hello,', decrypted)

    # with open('../../../data/dataset.pkl', 'rb') as f:
    #     _, __, ___, mfm_data = pickle.load(f)
    # with open('mfm.pkl', 'wb') as f:
    #     pickle.dump(mfm_data, f)

    # constants
    LOCATION = 'Location'
    MFLOWMTR = 'Mass Flow Meter'
    SORTED_COLUMNS = ['mflo' + str(i) for i in range(1, 7)] + ['vflo' + str(i) for i in range(1, 7)] + \
                     ['dens' + str(i) for i in range(1, 7)] + ['temp' + str(i) for i in range(1, 7)]
    PARAMS = ['Mass Flow', 'Volume Flow', 'Density', 'Temperature']
    WINDOW = 12

    # sidebar page
    st.sidebar.title('POSH Grace')
    page = st.sidebar.radio('Parameter', [LOCATION, MFLOWMTR, ''])
    # time frames
    ranges, bgn, end = get_time_frames()
    ranges = ['None selected'] + ranges
    seg = st.sidebar.selectbox('Time Frame', ranges)
    if seg != 'None selected':
        if ranges[0] == 'None selected':
            ranges = ranges[1:]
        seg = ranges.index(seg)
        bgn, end = bgn[seg], end[seg]
        span = ceil((end - bgn).days)
        if span > 1:
            window_num = list(range(1, span + 1))
            window_num = st.sidebar.radio('24-hour window', window_num, index=window_num[-1]-1)
            window_num = int(window_num) - 1
        else:
            window_num = 0
        window_0, window_1 = bgn + pd.Timedelta(WINDOW * window_num, unit='h'), bgn + pd.Timedelta(WINDOW * (window_num + 1), unit='h')
        # the rest of side bar is contextual

        # mainpage
        st.title('Efficient Ship')
        st.subheader('(work in progress)')
        # 
        if page == LOCATION:
            loc_data = load_locations(bgn, end)
            r = location(loc_data, window_0, window_1)
            if r:
                st.pydeck_chart(r)
            else:
                # st.write(r)
                st.subheader('No location data available')
        elif page == MFLOWMTR:
            mfm_data = load_mfm_data()
            mfm_slice = slice(mfm_data, bgn, end)
            if span > 1:
                to_plot = mfm_slice.loc[(mfm_slice.index >= window_0) & (mfm_slice.index <= window_1)]
            else:
                to_plot = mfm_slice
            to_plot = to_plot[SORTED_COLUMNS]
            to_plot = to_plot.fillna(0)
            params_radio = st.sidebar.radio('Mass Flow Meter Parameter', PARAMS)
            to_plot = to_plot[[['mflo', 'vflo', 'dens', 'temp'][PARAMS.index(params_radio)] + str(i) for i in range(1, 7)]]
            st.line_chart(to_plot)
            for p in to_plot.columns:
                x = np.arange(1, to_plot[p].shape[0] + 1) * 5
                y = to_plot[p].to_numpy()
                if 'flo' in p:
                    if x.shape[0] > 1 and y.shape[0] > 1:
                        st.write(p, 'area under curve:', int(round(auc(x, y))), 'kg' if p[0] is 'm' else 'liters')
                    else:
                        st.write('no data')
            # to_plot
            if st.sidebar.radio('View/Download', ['View', 'Download']) == 'Download':
                st.write(f"Right-click link(s) below and choose 'Save As' (or 'Save Link As...'): &lt;your_file_name&gt;.csv")
                for i in range(0, mfm_slice.shape[0], 5000):
                    csv = mfm_slice.iloc[i*5000:i*5000+5000].to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                    href = f'<a href="data:file/csv;base64,{b64}">Download CSV file part {1+i//5000}</a>'
                    st.markdown(href, unsafe_allow_html=True)
        else:
            # text = """\
            # There is currently (20191204) no official way of downloading data from Streamlit. See for
            # example [Issue 400](https://github.com/streamlit/streamlit/issues/400)

            # But I discovered a workaround
            # [here](https://github.com/holoviz/panel/issues/839#issuecomment-561538340).

            # It's based on the concept of
            # [HTML Data URLs](https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Data_URIs)

            # You can try it out below for a dataframe csv file download.

            # The methodology can be extended to other file types. For inspiration see
            # [base64.guru](https://base64.guru/converter/encode/file)
            # """
            # st.markdown(text)

            # data = [(1, 2, 3)]
            # # When no file name is given, pandas returns the CSV as a string, nice.
            # df = pd.DataFrame(data, columns=["Col1", "Col2", "Col3"])
            # csv = df.to_csv(index=False)
            # b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            # href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
            # st.markdown(href, unsafe_allow_html=True)
            pass
