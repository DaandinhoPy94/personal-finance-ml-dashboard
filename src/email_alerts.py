"""
Email Alert System voor Portfolio Changes
Automatische notificaties bij significante portfolio veranderingen
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
from typing import Dict, List

class PortfolioAlertSystem:
    """
    Automated email alerts voor portfolio performance
    """
    
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.alert_thresholds = {
            'major_change': 10.0,    # 10% change triggers alert
            'extreme_change': 25.0,  # 25% change triggers urgent alert
        }
        
    def setup_email_config(self, sender_email: str, sender_password: str, recipient_email: str):
        """
        Configureer email instellingen
        
        Args:
            sender_email: Gmail adres voor verzenden
            sender_password: App password (niet je normale wachtwoord!)
            recipient_email: Waar alerts naartoe moeten
        """
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
        
    def should_send_alert(self, current_stats: Dict, previous_stats: Dict = None) -> Dict:
        """
        Bepaalt of een alert verstuurd moet worden
        
        Args:
            current_stats: Huidige portfolio statistieken
            previous_stats: Vorige portfolio statistieken (optional)
            
        Returns:
            Dict met alert info (should_alert: bool, alert_type: str, etc.)
        """
        if not previous_stats:
            # Eerste keer - geen alert
            return {'should_alert': False, 'reason': 'No previous data'}
        
        current_value = current_stats.get('total_value_eur', 0)
        previous_value = previous_stats.get('total_value_eur', 0)
        
        if previous_value == 0:
            return {'should_alert': False, 'reason': 'Previous value was zero'}
        
        # Bereken percentage verandering
        change_pct = ((current_value - previous_value) / previous_value) * 100
        
        alert_info = {
            'should_alert': False,
            'change_pct': change_pct,
            'change_eur': current_value - previous_value,
            'current_value': current_value,
            'previous_value': previous_value,
            'alert_type': 'none'
        }
        
        # Check thresholds
        if abs(change_pct) >= self.alert_thresholds['extreme_change']:
            alert_info.update({
                'should_alert': True,
                'alert_type': 'extreme',
                'urgency': 'HIGH',
                'reason': f'Extreme portfolio change: {change_pct:+.1f}%'
            })
        elif abs(change_pct) >= self.alert_thresholds['major_change']:
            alert_info.update({
                'should_alert': True,
                'alert_type': 'major',
                'urgency': 'MEDIUM',
                'reason': f'Major portfolio change: {change_pct:+.1f}%'
            })
        
        return alert_info
    
    def generate_alert_email(self, alert_info: Dict, portfolio_stats: Dict) -> str:
        """
        Genereert HTML email content voor portfolio alert
        """
        change_pct = alert_info['change_pct']
        change_eur = alert_info['change_eur']
        current_value = alert_info['current_value']
        alert_type = alert_info['alert_type']
        
        # Determine colors and emojis
        if change_pct > 0:
            color = "#28a745"  # Green
            emoji = "📈"
            direction = "increased"
        else:
            color = "#dc3545"  # Red
            emoji = "📉"
            direction = "decreased"
        
        urgency_colors = {
            'HIGH': '#ff0000',
            'MEDIUM': '#ff8c00',
            'LOW': '#ffd700'
        }
        
        urgency_color = urgency_colors.get(alert_info.get('urgency', 'LOW'), '#ffd700')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: {urgency_color}; color: white; padding: 20px; border-radius: 10px; }}
                .content {{ padding: 20px; }}
                .metric {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .change {{ color: {color}; font-weight: bold; font-size: 1.2em; }}
                .holdings {{ margin-top: 20px; }}
                .footer {{ margin-top: 30px; padding: 15px; background-color: #e9ecef; border-radius: 5px; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{emoji} Portfolio Alert - {alert_type.upper()}</h1>
                <p>Your cryptocurrency portfolio has {direction} significantly!</p>
            </div>
            
            <div class="content">
                <div class="metric">
                    <h3>Portfolio Change</h3>
                    <p class="change">{change_pct:+.2f}% ({change_eur:+,.2f} EUR)</p>
                </div>
                
                <div class="metric">
                    <h3>Current Portfolio Value</h3>
                    <p><strong>€{current_value:,.2f}</strong></p>
                </div>
                
                <div class="metric">
                    <h3>24h Performance</h3>
                    <p>{portfolio_stats.get('total_change_24h', 0):+.2f}%</p>
                </div>
        """
        
        # Add holdings breakdown if available
        if 'holdings_detail' in portfolio_stats and portfolio_stats['holdings_detail']:
            html_content += """
                <div class="holdings">
                    <h3>Holdings Breakdown</h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr style="background-color: #f8f9fa;">
                            <th style="padding: 10px; border: 1px solid #ddd;">Coin</th>
                            <th style="padding: 10px; border: 1px solid #ddd;">Amount</th>
                            <th style="padding: 10px; border: 1px solid #ddd;">Value (EUR)</th>
                            <th style="padding: 10px; border: 1px solid #ddd;">24h Change</th>
                        </tr>
            """
            
            for holding in portfolio_stats['holdings_detail']:
                change_color = "#28a745" if holding['change_24h'] > 0 else "#dc3545"
                html_content += f"""
                        <tr>
                            <td style="padding: 10px; border: 1px solid #ddd;">{holding['symbol']}</td>
                            <td style="padding: 10px; border: 1px solid #ddd;">{holding['amount']:.4f}</td>
                            <td style="padding: 10px; border: 1px solid #ddd;">€{holding['value_eur']:,.2f}</td>
                            <td style="padding: 10px; border: 1px solid #ddd; color: {change_color};">{holding['change_24h']:+.1f}%</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
        
        html_content += f"""
            </div>
            
            <div class="footer">
                <p><strong>Alert Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Alert Type:</strong> {alert_type.upper()} ({alert_info.get('urgency', 'MEDIUM')} priority)</p>
                <p><em>This is an automated alert from your Personal Finance ML Dashboard.</em></p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def send_portfolio_alert(self, alert_info: Dict, portfolio_stats: Dict) -> bool:
        """
        Verstuurt portfolio alert email
        
        Returns:
            bool: True if email sent successfully
        """
        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = f"🚨 Portfolio Alert: {alert_info['change_pct']:+.1f}% Change"
            message["From"] = self.sender_email
            message["To"] = self.recipient_email
            
            # Generate HTML content
            html_content = self.generate_alert_email(alert_info, portfolio_stats)
            
            # Create HTML part
            html_part = MIMEText(html_content, "html")
            message.attach(html_part)
            
            # Create secure SSL connection and send
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.recipient_email, message.as_string())
            
            print(f"✅ Portfolio alert email sent successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send email: {str(e)}")
            return False
    
    def test_email_setup(self) -> bool:
        """
        Test email configuratie met simpele test email
        """
        try:
            # Simple test message
            message = MIMEText("Test email from your Personal Finance Dashboard! Email alerts are working properly. 🎉")
            message["Subject"] = "✅ Email Alert Test - Personal Finance Dashboard"
            message["From"] = self.sender_email
            message["To"] = self.recipient_email
            
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.recipient_email, message.as_string())
            
            print("✅ Test email sent successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Email test failed: {str(e)}")
            return False

# Example usage and testing
if __name__ == "__main__":
    print("📧 Email Alert System Test")
    print("Note: You need to configure email settings in the app first!")