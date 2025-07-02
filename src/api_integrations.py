"""
API Integrations voor live financial data
CoinGecko crypto prices, Yahoo Finance stocks, currency rates
"""

import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import asyncio
import time
import json
from typing import Dict, List, Optional

class CryptoDataFetcher:
    """
    Haalt live cryptocurrency data op van CoinGecko API
    Gratis, geen API key nodig, betrouwbare data
    """
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.supported_coins = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH', 
            'cardano': 'ADA',
            'solana': 'SOL',
            'polygon': 'MATIC',
            'chainlink': 'LINK',
            'litecoin': 'LTC',
            'dogecoin': 'DOGE'
        }
        
    def get_live_prices(self, coin_ids: List[str] = None) -> Dict:
        """
        Haalt live prijzen op voor crypto currencies
        
        Args:
            coin_ids: List van coin IDs (bijv. ['bitcoin', 'ethereum'])
            
        Returns:
            Dict met coin data: prices, market caps, changes
        """
        if coin_ids is None:
            coin_ids = list(self.supported_coins.keys())
        
        # CoinGecko API endpoint
        ids_string = ','.join(coin_ids)
        url = f"{self.base_url}/simple/price"
        
        params = {
            'ids': ids_string,
            'vs_currencies': 'eur,usd',
            'include_market_cap': 'true',
            'include_24hr_change': 'true',
            'include_last_updated_at': 'true'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()  # Raise exception voor HTTP errors
            
            data = response.json()
            
            # Transform naar user-friendly format
            processed_data = {}
            
            for coin_id, coin_data in data.items():
                processed_data[coin_id] = {
                    'symbol': self.supported_coins.get(coin_id, coin_id.upper()),
                    'name': coin_id.replace('-', ' ').title(),
                    'price_eur': coin_data.get('eur', 0),
                    'price_usd': coin_data.get('usd', 0),
                    'market_cap_eur': coin_data.get('eur_market_cap', 0),
                    'change_24h': coin_data.get('eur_24h_change', 0),
                    'last_updated': datetime.fromtimestamp(coin_data.get('last_updated_at', 0))
                }
            
            return processed_data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API Error: {e}")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ JSON Decode Error: {e}")
            return {}
    
    def get_historical_data(self, coin_id: str, days: int = 30) -> pd.DataFrame:
        """
        Haalt historische prijsdata op voor een cryptocurrency
        
        Args:
            coin_id: Coin identifier (bijv. 'bitcoin')
            days: Aantal dagen geschiedenis (max 365 voor gratis tier)
            
        Returns:
            DataFrame met datum, prijs, volume data
        """
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        
        params = {
            'vs_currency': 'eur',
            'days': days,
            'interval': 'daily' if days > 30 else 'hourly'
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract price data
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            # Convert naar DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['volume'] = [vol[1] for vol in volumes] if volumes else 0
            
            # Clean up
            df = df.drop('timestamp', axis=1)
            df = df.set_index('date')
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Historical Data Error: {e}")
            return pd.DataFrame()

class PortfolioTracker:
    """
    Tracked cryptocurrency portfolio waarde en performance
    """
    
    def __init__(self):
        self.crypto_fetcher = CryptoDataFetcher()
        self.holdings = {}  # Will store user's crypto holdings
        
    def set_holdings(self, holdings: Dict[str, float]):
        """
        Stelt crypto holdings in
        
        Args:
            holdings: Dict zoals {'bitcoin': 0.5, 'ethereum': 2.0}
        """
        self.holdings = holdings
        
    def calculate_portfolio_value(self) -> Dict:
        """
        Berekent totale portfolio waarde en individual holdings
        
        Returns:
            Dict met portfolio statistieken
        """
        if not self.holdings:
            return {'total_value_eur': 0, 'total_value_usd': 0, 'holdings_detail': []}
        
        # Get live prices for holdings
        coin_ids = list(self.holdings.keys())
        live_prices = self.crypto_fetcher.get_live_prices(coin_ids)
        
        portfolio_stats = {
            'total_value_eur': 0,
            'total_value_usd': 0,
            'total_change_24h': 0,
            'holdings_detail': [],
            'last_updated': datetime.now()
        }
        
        for coin_id, amount in self.holdings.items():
            if coin_id in live_prices:
                coin_data = live_prices[coin_id]
                
                value_eur = amount * coin_data['price_eur']
                value_usd = amount * coin_data['price_usd']
                change_24h = coin_data['change_24h']
                
                portfolio_stats['total_value_eur'] += value_eur
                portfolio_stats['total_value_usd'] += value_usd
                
                holding_detail = {
                    'coin': coin_data['name'],
                    'symbol': coin_data['symbol'],
                    'amount': amount,
                    'price_eur': coin_data['price_eur'],
                    'value_eur': value_eur,
                    'change_24h': change_24h,
                    'change_value_eur': value_eur * (change_24h / 100)
                }
                
                portfolio_stats['holdings_detail'].append(holding_detail)
        
        # Calculate weighted average change
        if portfolio_stats['total_value_eur'] > 0:
            weighted_change = sum(
                detail['change_value_eur'] for detail in portfolio_stats['holdings_detail']
            )
            portfolio_stats['total_change_24h'] = (weighted_change / portfolio_stats['total_value_eur']) * 100
        
        return portfolio_stats

class CurrencyConverter:
    """
    Real-time valutakoersen van exchangerate-api.com
    """
    
    def __init__(self):
        self.base_url = "https://api.exchangerate-api.com/v4/latest"
        
    def get_exchange_rates(self, base_currency: str = "EUR") -> Dict:
        """
        Haalt actuele wisselkoersen op
        
        Args:
            base_currency: Basis valuta (EUR, USD, GBP, etc.)
            
        Returns:
            Dict met exchange rates
        """
        try:
            url = f"{self.base_url}/{base_currency}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'base': data['base'],
                'rates': data['rates'],
                'last_updated': datetime.fromtimestamp(data['time_last_updated'])
            }
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Currency API Error: {e}")
            return {}
    
    def convert_amount(self, amount: float, from_currency: str, to_currency: str) -> float:
        """
        Converteerd bedrag tussen valuta's
        """
        rates = self.get_exchange_rates(from_currency)
        
        if to_currency in rates.get('rates', {}):
            return amount * rates['rates'][to_currency]
        
        return amount  # Return original if conversion fails

# Test functions
def test_crypto_api():
    """Test functie om API te checken"""
    fetcher = CryptoDataFetcher()
    
    print("🚀 Testing CoinGecko API...")
    prices = fetcher.get_live_prices(['bitcoin', 'ethereum'])
    
    if prices:
        for coin, data in prices.items():
            print(f"✅ {coin}: €{data['price_eur']:.2f} ({data['change_24h']:+.1f}%)")
    else:
        print("❌ API test failed")
    
    return len(prices) > 0

if __name__ == "__main__":
    # Quick test when running directly
    test_crypto_api()