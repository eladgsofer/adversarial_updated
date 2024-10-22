rom
psycopg2.pool
import SimpleConnectionPool
from typing import List, Dict




def get_user_events(user_name_list: List[str], events_list: List[Dict]) -> List[Dict]:
    """
    This method gets a LONG list of MANY user names, and a VERY LONG list of system events, and returns the system events that are related
    to those users.
    Note that the events_list input contains an ID from the DB, the user_name_list contains the user name from the DB
    Args:
        user_name_list: List of user names
        events_list: List of system events
    Returns:
        all events related to the given users
    """

    # TODO perform one query to get all the user names, and create a mapper between username->id
    usernames_lst = ','.join(user_name_list)
    user_lst = cur.execute(f"SELECT * FROM users WHERE name in '{usernames_lst}'")
    # fetch

    mapper = {user_row['user_name']: user_row['user_id'] for user_row in user_lst }

    table[table['user_id'].isin(mapepr.values())]

    # TODO iterate all the events (possible with converting events to pandas table as well to perform vectorization)

    related_events = []
    for user_name in user_name_list:
        conn = conn_pool.getconn()
        with conn:
            cur = conn.cursor()

            cur.execute(f"SELECT * FROM users WHERE name = '{user_name}'")
            user = cur.fetchone()
            user_id = str(user[0])

            user_events = [e for e in events_list if e['user_id'] == user_id]
            related_events.extend(user_events)

    return related_events


def test_get_user_events():
    user_names = ['Shlomo', 'Itzik']
    # In the DB, Shlomo is user ID 2 and Itzik is user ID 3
    events = [
        {
            "event_id": "17003212",
            "user_id": "1",
            "event_type": "login",
            "event_params": {
                "result": "success",
                "host": "1.2.3.4"
            }
        },
        {
            "event_id": "17003213",
            "user_id": "1",
            "event_type": "login",
            "event_params": {
                "result": "failure",
                "host": "11.12.13.14"
            }
        },
        {
            "event_id": "17003214",
            "user_id": "2",
            "event_type": "file_access",
            "event_params": {
                "result": "failure",
                "file_name": "/tmp/secret.txt"
            }
        },
        {
            "event_id": "17003215",
            "user_id": "3",
            "event_type": "file_access",
            "event_params": {
                "result": "success",
                "file_name": "/tmp/secret.txt"
            }
        },
    ]

    expected_results = [
        {
            "event_id": "17003214",
            "user_id": "2",
            "event_type": "file_access",
            "event_params": {
                "result": "failure",
                "file_name": "/tmp/secret.txt"
            }
        },
        {
            "event_id": "17003215",
            "user_id": "3",
            "event_type": "file_access",
            "event_params": {
                "result": "success",
                "file_name": "/tmp/secret.txt"
            }
        },
    ]

    results = get_user_events(user_names, events)

    for r in results:
        assert r["event_id"] in [e["event_id"] for e in expected_results]



if __name__ == '__main__':
    # TODO get it via encrypted file or a different way.
    DB_SETTINGS = {
        'user': 'izrael',
        'password': 'Aa123456',
        'host': 'db.company.cyesec.com',
        'database': 'company',
        'port': '5432',
    }

    conn_pool = SimpleConnectionPool(0, 1, **DB_SETTINGS)
    test_get_user_events()