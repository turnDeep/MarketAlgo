# dashboard_google_sheets.py
"""
Google Sheets Dashboard Writer
ダッシュボードデータをGoogleスプレッドシートに出力
"""

import pandas as pd
from typing import Dict, Optional
from datetime import datetime
import logging

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    logging.warning("gspread not installed. Install with: pip install gspread google-auth")

logger = logging.getLogger(__name__)


class GoogleSheetsWriter:
    """Google Sheets Dashboard Writer"""

    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    def __init__(self, credentials_file: str = 'credentials.json', spreadsheet_name: str = 'Market Dashboard'):
        """
        Initialize Google Sheets Writer

        Args:
            credentials_file: Path to service account credentials JSON file
            spreadsheet_name: Name of the Google Spreadsheet
        """
        if not GSPREAD_AVAILABLE:
            raise ImportError("gspread is not installed. Install with: pip install gspread google-auth")

        self.credentials_file = credentials_file
        self.spreadsheet_name = spreadsheet_name
        self.client = None
        self.spreadsheet = None

    def authenticate(self) -> bool:
        """
        Authenticate with Google Sheets API

        Returns:
            True if successful, False otherwise
        """
        try:
            creds = Credentials.from_service_account_file(
                self.credentials_file,
                scopes=self.SCOPES
            )
            self.client = gspread.authorize(creds)
            logger.info("✓ Authenticated with Google Sheets API")
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate: {e}")
            return False

    def open_spreadsheet(self) -> bool:
        """
        Open or create the spreadsheet

        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to open existing spreadsheet
            self.spreadsheet = self.client.open(self.spreadsheet_name)
            logger.info(f"✓ Opened existing spreadsheet: {self.spreadsheet_name}")
            return True
        except gspread.SpreadsheetNotFound:
            # Create new spreadsheet
            try:
                self.spreadsheet = self.client.create(self.spreadsheet_name)
                logger.info(f"✓ Created new spreadsheet: {self.spreadsheet_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to create spreadsheet: {e}")
                return False
        except Exception as e:
            logger.error(f"Error opening spreadsheet: {e}")
            return False

    def write_dashboard_data(
        self,
        exposure: Dict,
        market_performance: pd.DataFrame,
        sectors_performance: pd.DataFrame,
        macro_performance: pd.DataFrame,
        screener_results: Dict,
        factors_vs_sp500: Dict = None,
        bond_yields: Dict = None,
        power_trend: Dict = None
    ):
        """
        Write dashboard data to Google Sheets

        Args:
            exposure: Market exposure data
            market_performance: Market performance DataFrame
            sectors_performance: Sectors performance DataFrame
            macro_performance: Macro performance DataFrame
            screener_results: Screener results dictionary
            factors_vs_sp500: Factors vs SP500 data
            bond_yields: Bond yields data
            power_trend: Power trend data
        """
        if not self.client or not self.spreadsheet:
            if not self.authenticate() or not self.open_spreadsheet():
                logger.error("Cannot write data - authentication or spreadsheet open failed")
                return

        try:
            # Write Summary sheet
            self._write_summary_sheet(exposure, factors_vs_sp500, bond_yields, power_trend)

            # Write Market Performance sheet
            self._write_performance_sheet('Market Performance', market_performance)

            # Write Sectors Performance sheet
            self._write_performance_sheet('Sectors Performance', sectors_performance)

            # Write Macro Performance sheet
            self._write_performance_sheet('Macro Performance', macro_performance)

            # Write Screener Results sheets
            self._write_screener_sheets(screener_results)

            logger.info("✓ All data written to Google Sheets successfully")
            logger.info(f"📊 Spreadsheet URL: {self.spreadsheet.url}")

        except Exception as e:
            logger.error(f"Error writing to Google Sheets: {e}", exc_info=True)

    def _write_summary_sheet(
        self,
        exposure: Dict,
        factors_vs_sp500: Dict,
        bond_yields: Dict,
        power_trend: Dict
    ):
        """Write Summary sheet with key metrics"""
        try:
            # Get or create worksheet
            try:
                worksheet = self.spreadsheet.worksheet('Summary')
                worksheet.clear()
            except gspread.WorksheetNotFound:
                worksheet = self.spreadsheet.add_worksheet('Summary', rows=100, cols=20)

            # Prepare data
            data = []
            data.append(['Market Dashboard Summary'])
            data.append(['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            data.append([])

            # Market Exposure
            data.append(['MARKET EXPOSURE'])
            data.append(['Score:', f"{exposure.get('score', 0):.2f}%"])
            data.append(['Level:', exposure.get('level', 'N/A')])
            data.append(['Positive Factors:', f"{exposure.get('positive_count', 0)}/{exposure.get('total_factors', 12)}"])
            data.append(['VIX Level:', f"{exposure.get('vix_level', 0):.2f}" if exposure.get('vix_level') else 'N/A'])
            data.append([])

            # Factors vs SP500
            if factors_vs_sp500:
                data.append(['FACTORS VS SP500 (Yesterday)'])
                for name, value in factors_vs_sp500.items():
                    data.append([name, f"{value:+.2f}%"])
                data.append([])

            # Bond Yields
            if bond_yields:
                data.append(['BOND YIELDS'])
                for name, value in bond_yields.items():
                    data.append([name, f"{value:.2f}%"])
                data.append([])

            # Power Trend
            if power_trend:
                data.append(['POWER TREND'])
                data.append(['RSI:', f"{power_trend.get('rsi', 0):.2f}"])
                data.append(['MACD Histogram:', f"{power_trend.get('macd_histogram', 0):.2f}"])
                data.append(['Trend:', power_trend.get('trend', 'N/A')])

            # Write data
            worksheet.update('A1', data)

            # Format header
            worksheet.format('A1', {
                'textFormat': {'bold': True, 'fontSize': 14},
                'backgroundColor': {'red': 0.4, 'green': 0.5, 'blue': 0.8}
            })

            logger.info("✓ Summary sheet updated")

        except Exception as e:
            logger.error(f"Error writing summary sheet: {e}")

    def _write_performance_sheet(self, sheet_name: str, df: pd.DataFrame):
        """Write performance data to a sheet"""
        try:
            if df.empty:
                logger.warning(f"No data for {sheet_name}")
                return

            # Get or create worksheet
            try:
                worksheet = self.spreadsheet.worksheet(sheet_name)
                worksheet.clear()
            except gspread.WorksheetNotFound:
                worksheet = self.spreadsheet.add_worksheet(sheet_name, rows=100, cols=20)

            # Prepare data
            data = []
            data.append([sheet_name])
            data.append([])

            # Convert DataFrame to list of lists
            headers = df.columns.tolist()
            data.append(headers)

            for _, row in df.iterrows():
                row_data = []
                for col in headers:
                    value = row[col]
                    if pd.isna(value):
                        row_data.append('')
                    elif isinstance(value, (int, float)):
                        if col in ['ticker', 'index']:
                            row_data.append(str(value))
                        else:
                            row_data.append(value)
                    elif isinstance(value, bool):
                        row_data.append('✓' if value else '✗')
                    elif isinstance(value, list):
                        row_data.append(str(value))
                    else:
                        row_data.append(str(value))
                data.append(row_data)

            # Write data
            worksheet.update('A1', data)

            # Format header
            worksheet.format('A1', {
                'textFormat': {'bold': True, 'fontSize': 14},
                'backgroundColor': {'red': 0.4, 'green': 0.5, 'blue': 0.8}
            })
            worksheet.format('A3', {
                'textFormat': {'bold': True},
                'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
            })

            logger.info(f"✓ {sheet_name} sheet updated")

        except Exception as e:
            logger.error(f"Error writing {sheet_name} sheet: {e}")

    def _write_screener_sheets(self, screener_results: Dict):
        """Write screener results to separate sheets"""
        screener_names = {
            'momentum_97': 'Momentum 97',
            'explosive_eps': 'Explosive EPS Growth',
            'healthy_chart': 'Healthy Chart',
            'up_on_volume': 'Up on Volume',
            'top_2_rs': 'Top 2% RS Rating',
            'bullish_4pct': '4% Bullish Yesterday'
        }

        for key, name in screener_names.items():
            df = screener_results.get(key, pd.DataFrame())
            if not df.empty:
                # Limit to first 50 rows
                df_limited = df.head(50)
                self._write_performance_sheet(name, df_limited)


def write_to_google_sheets(
    credentials_file: str,
    spreadsheet_name: str,
    exposure: Dict,
    market_performance: pd.DataFrame,
    sectors_performance: pd.DataFrame,
    macro_performance: pd.DataFrame,
    screener_results: Dict,
    factors_vs_sp500: Dict = None,
    bond_yields: Dict = None,
    power_trend: Dict = None
) -> bool:
    """
    Convenience function to write dashboard data to Google Sheets

    Returns:
        True if successful, False otherwise
    """
    try:
        writer = GoogleSheetsWriter(credentials_file, spreadsheet_name)
        if not writer.authenticate():
            return False
        if not writer.open_spreadsheet():
            return False

        writer.write_dashboard_data(
            exposure,
            market_performance,
            sectors_performance,
            macro_performance,
            screener_results,
            factors_vs_sp500,
            bond_yields,
            power_trend
        )

        return True
    except Exception as e:
        logger.error(f"Failed to write to Google Sheets: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    # Test
    print("Testing Google Sheets Writer...")

    # Create test data
    test_exposure = {
        'score': 75.0,
        'level': 'Positive',
        'positive_count': 9,
        'total_factors': 12,
        'vix_level': 15.5
    }

    test_market_df = pd.DataFrame({
        'ticker': ['SPY', 'QQQ'],
        'price': [450.0, 380.0],
        '% 1D': [0.5, 0.8],
        'RS STS %': [85, 90]
    })

    test_sectors_df = pd.DataFrame({
        'ticker': ['XLK', 'XLF'],
        'price': [180.0, 38.0],
        '% 1D': [1.2, -0.3],
        'RS STS %': [88, 75]
    })

    test_macro_df = pd.DataFrame({
        'ticker': ['^VIX', 'TLT'],
        'price': [15.5, 95.0],
        '% 1D': [-2.0, 0.1]
    })

    test_screeners = {
        'momentum_97': pd.DataFrame({
            'ticker': ['NVDA', 'AAPL'],
            'returns_1m': [15.0, 10.0],
            'returns_3m': [25.0, 18.0]
        })
    }

    # Write to Google Sheets
    success = write_to_google_sheets(
        credentials_file='credentials.json',
        spreadsheet_name='Market Dashboard Test',
        exposure=test_exposure,
        market_performance=test_market_df,
        sectors_performance=test_sectors_df,
        macro_performance=test_macro_df,
        screener_results=test_screeners
    )

    if success:
        print("✓ Test completed successfully")
    else:
        print("✗ Test failed")
