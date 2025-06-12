import yfinance as yf
import pandas as pd
import os

def clear_screen():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_stock_data(ticker):
    """Fetches real-time stock data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.info  # Use info for a more concise set of data
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def display_portfolio(portfolio):
    """Displays the current stock portfolio."""
    clear_screen()
    if not portfolio:
        print("Your portfolio is empty.")
        return

    print("\n--- Your Portfolio ---")
    df = pd.DataFrame(portfolio).T  # Transpose for better readability
    print(df[['shortName', 'currentPrice', 'quantity', 'previousClose', 'marketCap', 'sector']].to_string())
    print("-" * 20)

def main():
    """Main function to run the stock portfolio tracker."""
    portfolio = {}

    while True:
        clear_screen()
        print("\nStock Portfolio Tracker")
        print("1. Add Stock")
        print("2. Remove Stock")
        print("3. View Portfolio")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            ticker = input("Enter stock ticker (e.g., AAPL): ").upper()
            quantity = int(input("Enter quantity: "))
            stock_data = get_stock_data(ticker)

            if stock_data:
                if ticker in portfolio:
                    portfolio[ticker]['quantity'] += quantity
                else:
                    portfolio[ticker] = stock_data
                    portfolio[ticker]['quantity'] = quantity
                print(f"{ticker} added to portfolio.")
            input("Press Enter to continue...")

        elif choice == '2':
            ticker = input("Enter stock ticker to remove: ").upper()
            if ticker in portfolio:
                del portfolio[ticker]
                print(f"{ticker} removed from portfolio.")
            else:
                print(f"{ticker} not found in portfolio.")
            input("Press Enter to continue...")

        elif choice == '3':
            display_portfolio(portfolio)
            input("Press Enter to continue...")

        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()