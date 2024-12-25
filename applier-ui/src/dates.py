import calendar
from datetime import datetime, timedelta


def get_previous_month_name(n: int):
    # Get the current date
    today = datetime.today()
    
    # Calculate the target date n months back
    for _ in range(n):
        # Subtract one month at a time
        first_day_of_current_month = today.replace(day=1)
        today = first_day_of_current_month - timedelta(days=1)
    
    # Return the month name
    return today.strftime('%B %Y')
