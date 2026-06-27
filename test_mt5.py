import MetaTrader5 as mt5

print(">>> KÄYNNISTETÄÄN MT5 DIAGNOSTIIKKA <<<")

# 1. Alustus
if not mt5.initialize():
    print("🔴 VIRHE: MT5 alustus epäonnistui.")
    quit()

info = mt5.account_info()
if info is None:
    print("🔴 VIRHE: Ei yhteyttä tiliin. Oletko kirjautunut sisään MT5:ssä?")
    mt5.shutdown()
    quit()

print(f"🟢 Yhdistetty: {info.company} (Tili: {info.login})")

# 2. Testataan yhtä symbolia (Vaihda tähän tarkka välittäjän käyttämä nimi, esim. "EURUSD" tai "EURUSD.raw")
TEST_SYMBOL = "EURUSD" 

# Yritetään pakottaa symboli Market Watch -ikkunaan
is_selected = mt5.symbol_select(TEST_SYMBOL, True)
if not is_selected:
    print(f"🔴 VIRHE: Välittäjä ei salli symbolin '{TEST_SYMBOL}' valintaa. Nimi on varmasti väärin tai markkina on suljettu.")
else:
    print(f"🟢 Symboli '{TEST_SYMBOL}' löydetty ja aktivoitu.")

# 3. Yritetään hakea Tick-data (Raaka fysiikka)
tick = mt5.symbol_info_tick(TEST_SYMBOL)

if tick is None:
    print(f"🔴 VIRHE: Symboli löytyi, mutta Tick-dataa ei tule. Syitä voivat olla:")
    print("   1. Markkina on juuri nyt kiinni (Tarkista MT5:stä liikkuuko hinta).")
    print("   2. Välittäjä estää datan luvun API:n kautta.")
else:
    print("🟢 MENESTYS! Data virtaa:")
    print(f"   - Hinta (Last): {tick.last}")
    print(f"   - Bid/Ask: {tick.bid} / {tick.ask}")
    print(f"   - Volyymi: {tick.volume}")
    print(f"   - Aika: {tick.time_msc} ms")

mt5.shutdown()
print(">>> DIAGNOSTIIKKA PÄÄTTYNYT <<<")