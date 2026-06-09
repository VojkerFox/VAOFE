import requests

def send_telegram_message(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("✅ Viesti lähetetty onnistuneesti Telegramiin!")
        else:
            print(f"❌ Virhe lähetettäessä: {response.text}")
    except Exception as e:
        print(f"❌ Yhteysvirhe: {e}")

if __name__ == "__main__":
    BOT_TOKEN = "8658806596:AAH3jFlP7LKuHY8wMXBt02kD9UMC9SacZRI"
    CHAT_ID = "260783230"
    TESTIVIESTI = "🚀 VAOFE-moottori ilmoittautuu! Dataputki on auki ja viestit kulkevat perille asti."
    
    send_telegram_message(BOT_TOKEN, CHAT_ID, TESTIVIESTI)