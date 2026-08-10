import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random
from typing import Dict, List, Tuple, Optional
import holidays
import logging

logger = logging.getLogger(__name__)


class RealisticSalesDataGenerator:
    """Generate realistic sales data with multiple files, partitions, and business patterns"""
    
    def __init__(self, start_date: str = "2022-01-01", end_date: str = "2023-12-31"):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.us_holidays = holidays.US()
        
        # Store configurations
        self.stores = {
            'store_001': {'location': 'New York', 'size': 'large', 'base_traffic': 1000},
            'store_002': {'location': 'Los Angeles', 'size': 'large', 'base_traffic': 950},
            'store_003': {'location': 'Chicago', 'size': 'medium', 'base_traffic': 700},
            'store_004': {'location': 'Houston', 'size': 'medium', 'base_traffic': 650},
            'store_005': {'location': 'Phoenix', 'size': 'small', 'base_traffic': 400},
            'store_006': {'location': 'Philadelphia', 'size': 'medium', 'base_traffic': 600},
            'store_007': {'location': 'San Antonio', 'size': 'small', 'base_traffic': 350},
            'store_008': {'location': 'San Diego', 'size': 'medium', 'base_traffic': 550},
            'store_009': {'location': 'Dallas', 'size': 'large', 'base_traffic': 850},
            'store_010': {'location': 'Miami', 'size': 'medium', 'base_traffic': 600}
        }
        
        # Product categories and items
        self.product_categories = {
            'Electronics': {
                'ELEC_001': {'name': 'Smartphone', 'price': 699, 'margin': 0.15, 'seasonality': 'holiday'},
                'ELEC_002': {'name': 'Laptop', 'price': 999, 'margin': 0.12, 'seasonality': 'back_to_school'},
                'ELEC_003': {'name': 'Headphones', 'price': 199, 'margin': 0.25, 'seasonality': 'holiday'},
                'ELEC_004': {'name': 'Tablet', 'price': 499, 'margin': 0.18, 'seasonality': 'holiday'},
                'ELEC_005': {'name': 'Smart Watch', 'price': 299, 'margin': 0.20, 'seasonality': 'fitness'}
            },
            'Clothing': {
                'CLTH_001': {'name': 'T-Shirt', 'price': 29, 'margin': 0.50, 'seasonality': 'summer'},
                'CLTH_002': {'name': 'Jeans', 'price': 79, 'margin': 0.45, 'seasonality': 'all_year'},
                'CLTH_003': {'name': 'Jacket', 'price': 149, 'margin': 0.40, 'seasonality': 'winter'},
                'CLTH_004': {'name': 'Dress', 'price': 89, 'margin': 0.48, 'seasonality': 'summer'},
                'CLTH_005': {'name': 'Shoes', 'price': 119, 'margin': 0.42, 'seasonality': 'all_year'}
            },
            'Home': {
                'HOME_001': {'name': 'Coffee Maker', 'price': 79, 'margin': 0.30, 'seasonality': 'holiday'},
                'HOME_002': {'name': 'Blender', 'price': 49, 'margin': 0.35, 'seasonality': 'summer'},
                'HOME_003': {'name': 'Vacuum Cleaner', 'price': 199, 'margin': 0.28, 'seasonality': 'spring'},
                'HOME_004': {'name': 'Air Purifier', 'price': 149, 'margin': 0.32, 'seasonality': 'all_year'},
                'HOME_005': {'name': 'Toaster', 'price': 39, 'margin': 0.40, 'seasonality': 'holiday'}
            },
            'Sports': {
                'SPRT_001': {'name': 'Yoga Mat', 'price': 29, 'margin': 0.55, 'seasonality': 'fitness'},
                'SPRT_002': {'name': 'Dumbbells', 'price': 49, 'margin': 0.45, 'seasonality': 'fitness'},
                'SPRT_003': {'name': 'Running Shoes', 'price': 129, 'margin': 0.38, 'seasonality': 'spring'},
                'SPRT_004': {'name': 'Bicycle', 'price': 399, 'margin': 0.25, 'seasonality': 'summer'},
                'SPRT_005': {'name': 'Tennis Racket', 'price': 89, 'margin': 0.35, 'seasonality': 'summer'}
            }
        }
        
        # Flatten products
        self.all_products = {}
        for category, products in self.product_categories.items():
            for product_id, product_info in products.items():
                self.all_products[product_id] = {**product_info, 'category': category}
    
    def get_seasonality_factor(self, date: pd.Timestamp, seasonality_type: str) -> float:
        """Calculate seasonality factor based on date and type"""
        day_of_year = date.dayofyear
        
        if seasonality_type == 'holiday':
            # Peak during November-December and around major holidays
            if date.month in [11, 12]:
                return 1.5 + 0.3 * np.sin(2 * np.pi * (day_of_year - 300) / 60)
            elif date in self.us_holidays:
                return 1.3
            else:
                return 1.0
        
        elif seasonality_type == 'summer':
            # Peak June-August
            return 1.0 + 0.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        elif seasonality_type == 'winter':
            # Peak December-February
            return 1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year + 90) / 365)
        
        elif seasonality_type == 'back_to_school':
            # Peak August-September
            if date.month in [8, 9]:
                return 1.4
            else:
                return 0.9
        
        elif seasonality_type == 'fitness':
            # Peak January (New Year) and May-June (summer prep)
            if date.month == 1:
                return 1.5
            elif date.month in [5, 6]:
                return 1.3
            else:
                return 1.0
        
        elif seasonality_type == 'spring':
            # Peak March-May
            return 1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year - 20) / 365)
        
        else:  # all_year
            return 1.0
    
    def get_day_of_week_factor(self, date: pd.Timestamp) -> float:
        """Get multiplier based on day of week"""
        dow = date.dayofweek
        # Monday=0, Sunday=6
        dow_factors = [0.9, 0.85, 0.85, 0.9, 1.1, 1.3, 1.2]
        return dow_factors[dow]
    
    def generate_promotions(self) -> pd.DataFrame:
        """Generate promotional calendar"""
        promotions = []
        
        # Major sales events
        major_events = [
            ('Black Friday', 11, 4, 5, 0.25),  # 4th Friday of November, 5 days, 25% off
            ('Cyber Monday', 11, 4, 2, 0.20),  # Monday after Black Friday
            ('Christmas Sale', 12, 15, 10, 0.15),
            ('New Year Sale', 1, 1, 7, 0.20),
            ('Presidents Day', 2, 15, 3, 0.15),
            ('Memorial Day', 5, 25, 3, 0.15),
            ('July 4th Sale', 7, 1, 5, 0.15),
            ('Labor Day', 9, 1, 3, 0.15),
            ('Back to School', 8, 1, 14, 0.10),
        ]
        
        current_date = self.start_date
        while current_date <= self.end_date:
            year = current_date.year
            
            for event_name, month, day, duration, discount in major_events:
                if event_name == 'Black Friday':
                    # Calculate 4th Thursday of November, then add 1 for Friday
                    november = pd.Timestamp(year, 11, 1)
                    thursdays = pd.date_range(november, november + timedelta(days=30), freq='W-THU')
                    event_date = thursdays[3] + timedelta(days=1)
                else:
                    try:
                        event_date = pd.Timestamp(year, month, day)
                    except:
                        continue
                
                if self.start_date <= event_date <= self.end_date:
                    for d in range(duration):
                        promo_date = event_date + timedelta(days=d)
                        if promo_date <= self.end_date:
                            # Random products on promotion
                            promo_products = random.sample(list(self.all_products.keys()), 
                                                         k=random.randint(5, 15))
                            for product_id in promo_products:
                                promotions.append({
                                    'date': promo_date,
                                    'product_id': product_id,
                                    'promotion_type': event_name,
                                    'discount_percent': discount
                                })
            
            current_date = current_date + pd.DateOffset(years=1)
        
        # Add random flash sales
        n_flash_sales = int((self.end_date - self.start_date).days * 0.05)  # 5% of days
        flash_dates = pd.date_range(self.start_date, self.end_date, periods=n_flash_sales)
        
        for date in flash_dates:
            promo_products = random.sample(list(self.all_products.keys()), k=random.randint(3, 8))
            for product_id in promo_products:
                promotions.append({
                    'date': date,
                    'product_id': product_id,
                    'promotion_type': 'Flash Sale',
                    'discount_percent': random.uniform(0.1, 0.3)
                })
        
        return pd.DataFrame(promotions)
    
    def generate_store_events(self) -> pd.DataFrame:
        """Generate store-specific events (closures, renovations, etc.)"""
        events = []
        
        for store_id, store_info in self.stores.items():
            # Random store closures (weather, technical issues)
            n_closures = random.randint(2, 5)
            closure_dates = pd.date_range(self.start_date, self.end_date, periods=n_closures)
            
            for date in closure_dates:
                events.append({
                    'store_id': store_id,
                    'date': date,
                    'event_type': 'closure',
                    'impact': -1.0  # 100% reduction
                })
            
            # Store renovations (longer impact)
            if random.random() < 0.3:  # 30% chance of renovation
                renovation_start = self.start_date + timedelta(days=random.randint(100, 600))
                renovation_duration = random.randint(7, 21)
                
                for d in range(renovation_duration):
                    reno_date = renovation_start + timedelta(days=d)
                    if reno_date <= self.end_date:
                        events.append({
                            'store_id': store_id,
                            'date': reno_date,
                            'event_type': 'renovation',
                            'impact': -0.3  # 30% reduction
                        })
        
        return pd.DataFrame(events)
    
    def generate_sales_data(self, output_dir: str = "/tmp/sales_data") -> Dict[str, List[str]]:
        """Generate realistic sales data partitioned by date and store (daily files)"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate supplementary data
        promotions_df = self.generate_promotions()
        store_events_df = self.generate_store_events()
        
        # Track file paths
        file_paths = {
            'sales': [],
            'inventory': [],
            'customer_traffic': [],
            'promotions': [],
            'store_events': []
        }
        
        # Save supplementary data
        promotions_path = os.path.join(output_dir, "promotions/promotions.parquet")
        os.makedirs(os.path.dirname(promotions_path), exist_ok=True)
        promotions_df.to_parquet(promotions_path, index=False)
        file_paths['promotions'].append(promotions_path)
        
        events_path = os.path.join(output_dir, "store_events/events.parquet")
        os.makedirs(os.path.dirname(events_path), exist_ok=True)
        store_events_df.to_parquet(events_path, index=False)
        file_paths['store_events'].append(events_path)
        
        # Generate sales data by day (more realistic for production)
        current_date = self.start_date
        
        while current_date <= self.end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            logger.info(f"Generating data for {date_str}")
            
            # Daily sales data for all stores
            daily_sales_data = []
            daily_traffic_data = []
            daily_inventory_data = []
            
            # Generate data for each store for this specific day
            for store_id, store_info in self.stores.items():
                # Store-level factors
                base_traffic = store_info['base_traffic']
                
                # Date factors
                dow_factor = self.get_day_of_week_factor(current_date)
                is_holiday = current_date in self.us_holidays
                holiday_factor = 1.3 if is_holiday else 1.0
                
                # Weather impact (random)
                weather_factor = np.random.normal(1.0, 0.1)
                weather_factor = max(0.5, min(1.2, weather_factor))
                
                # Check for store events
                store_event_impact = 1.0
                if not store_events_df.empty:
                    event = store_events_df[
                        (store_events_df['store_id'] == store_id) & 
                        (store_events_df['date'] == current_date)
                    ]
                    if not event.empty:
                        store_event_impact = 1.0 + event.iloc[0]['impact']
                
                # Calculate store traffic
                store_traffic = int(
                    base_traffic * dow_factor * holiday_factor * 
                    weather_factor * store_event_impact * 
                    np.random.normal(1.0, 0.05)
                )
                
                daily_traffic_data.append({
                    'date': current_date,
                    'store_id': store_id,
                    'customer_traffic': store_traffic,
                    'weather_impact': weather_factor,
                    'is_holiday': is_holiday
                })
                
                # Generate product-level sales
                for product_id, product_info in self.all_products.items():
                    # Product seasonality
                    seasonality_factor = self.get_seasonality_factor(
                        current_date, product_info['seasonality']
                    )
                    
                    # Check for promotions
                    promotion_factor = 1.0
                    discount_percent = 0.0
                    if not promotions_df.empty:
                        promo = promotions_df[
                            (promotions_df['date'] == current_date) & 
                            (promotions_df['product_id'] == product_id)
                        ]
                        if not promo.empty:
                            discount_percent = promo.iloc[0]['discount_percent']
                            # Promotion increases demand
                            promotion_factor = 1.0 + (discount_percent * 3)  # 3x multiplier
                    
                    # Calculate sales quantity
                    # Base conversion rate depends on store size and product price
                    size_factor = {'large': 1.0, 'medium': 0.7, 'small': 0.5}[store_info['size']]
                    price_factor = 1.0 / (1.0 + product_info['price'] / 100)  # Higher price, lower volume
                    
                    base_quantity = store_traffic * 0.001 * size_factor * price_factor
                    
                    quantity = int(
                        base_quantity * seasonality_factor * promotion_factor *
                        np.random.normal(1.0, 0.2)
                    )
                    quantity = max(0, quantity)
                    
                    # Calculate revenue
                    actual_price = product_info['price'] * (1 - discount_percent)
                    revenue = quantity * actual_price
                    cost = quantity * product_info['price'] * (1 - product_info['margin'])
                    
                    if quantity > 0:
                        daily_sales_data.append({
                            'date': current_date,
                            'store_id': store_id,
                            'product_id': product_id,
                            'category': product_info['category'],
                            'quantity_sold': quantity,
                            'unit_price': product_info['price'],
                            'discount_percent': discount_percent,
                            'revenue': revenue,
                            'cost': cost,
                            'profit': revenue - cost
                        })
                    
                    # Inventory tracking
                    inventory_level = random.randint(50, 200)
                    reorder_point = random.randint(20, 50)
                    
                    daily_inventory_data.append({
                        'date': current_date,
                        'store_id': store_id,
                        'product_id': product_id,
                        'inventory_level': inventory_level,
                        'reorder_point': reorder_point,
                        'days_of_supply': inventory_level / max(1, quantity)
                    })
            
            # Save daily files with proper partitioning
            # Sales data - one file per day
            if daily_sales_data:
                sales_df = pd.DataFrame(daily_sales_data)
                sales_path = os.path.join(
                    output_dir, 
                    f"sales/year={current_date.year}/month={current_date.month:02d}/day={current_date.day:02d}/"
                    f"sales_{date_str}.parquet"
                )
                os.makedirs(os.path.dirname(sales_path), exist_ok=True)
                sales_df.to_parquet(sales_path, index=False)
                file_paths['sales'].append(sales_path)
            
            # Customer traffic data - one file per day
            if daily_traffic_data:
                traffic_df = pd.DataFrame(daily_traffic_data)
                traffic_path = os.path.join(
                    output_dir,
                    f"customer_traffic/year={current_date.year}/month={current_date.month:02d}/day={current_date.day:02d}/"
                    f"traffic_{date_str}.parquet"
                )
                os.makedirs(os.path.dirname(traffic_path), exist_ok=True)
                traffic_df.to_parquet(traffic_path, index=False)
                file_paths['customer_traffic'].append(traffic_path)
            
            # Inventory data - daily snapshots
            if daily_inventory_data and current_date.dayofweek == 6:  # Weekly on Sundays
                inventory_df = pd.DataFrame(daily_inventory_data)
                inventory_path = os.path.join(
                    output_dir,
                    f"inventory/year={current_date.year}/week={current_date.isocalendar()[1]:02d}/"
                    f"inventory_{date_str}.parquet"
                )
                os.makedirs(os.path.dirname(inventory_path), exist_ok=True)
                inventory_df.to_parquet(inventory_path, index=False)
                file_paths['inventory'].append(inventory_path)
            
            # Move to next day
            current_date = current_date + timedelta(days=1)
        
        # Generate metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'n_stores': len(self.stores),
            'n_products': len(self.all_products),
            'file_counts': {k: len(v) for k, v in file_paths.items()},
            'total_files': sum(len(v) for v in file_paths.values())
        }
        
        metadata_df = pd.DataFrame([metadata])
        metadata_path = os.path.join(output_dir, "metadata/generation_metadata.parquet")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        metadata_df.to_parquet(metadata_path, index=False)
        
        logger.info(f"Generated {metadata['total_files']} files")
        logger.info(f"Sales files: {len(file_paths['sales'])}")
        logger.info(f"Output directory: {output_dir}")
        
        return file_paths


class ContactCenterDemandGenerator:
    """
    Generate synthetic contact-center operational demand data.

    Produces hourly records with timestamp, business unit, channel, volume,
    average handle time (AHT), day-of-week, hour-of-day, holiday indicators,
    and realistic intraday / weekly / seasonal demand patterns.
    """

    def __init__(
        self,
        start_date: str = "2022-01-01",
        end_date: str = "2023-12-31",
        seed: Optional[int] = 42,
    ):
        self.start_date = pd.to_datetime(start_date).normalize()
        self.end_date = pd.to_datetime(end_date).normalize()
        self.us_holidays = holidays.US()
        self.rng = np.random.default_rng(seed)

        # Business units with baseline hourly offered volume and AHT (seconds)
        self.business_units = {
            "sales_support": {
                "base_volume": 42,
                "base_aht_seconds": 320,
                "seasonality": "retail",
            },
            "billing": {
                "base_volume": 55,
                "base_aht_seconds": 420,
                "seasonality": "billing",
            },
            "technical_support": {
                "base_volume": 70,
                "base_aht_seconds": 540,
                "seasonality": "tech",
            },
            "retention": {
                "base_volume": 28,
                "base_aht_seconds": 600,
                "seasonality": "retention",
            },
            "general_inquiry": {
                "base_volume": 35,
                "base_aht_seconds": 240,
                "seasonality": "all_year",
            },
        }

        # Channel mix + relative volume / AHT multipliers
        self.channels = {
            "voice": {"volume_share": 0.55, "aht_multiplier": 1.00, "weekend_factor": 0.55},
            "chat": {"volume_share": 0.25, "aht_multiplier": 0.55, "weekend_factor": 0.85},
            "email": {"volume_share": 0.15, "aht_multiplier": 1.40, "weekend_factor": 0.70},
            "social": {"volume_share": 0.05, "aht_multiplier": 0.70, "weekend_factor": 1.10},
        }

        # Typical contact-center intraday profile (hour -> multiplier)
        self.hour_of_day_profile = {
            0: 0.08, 1: 0.05, 2: 0.04, 3: 0.04, 4: 0.05, 5: 0.10,
            6: 0.25, 7: 0.55, 8: 0.85, 9: 1.20, 10: 1.35, 11: 1.25,
            12: 0.95, 13: 1.15, 14: 1.30, 15: 1.25, 16: 1.10, 17: 0.90,
            18: 0.70, 19: 0.55, 20: 0.40, 21: 0.28, 22: 0.18, 23: 0.12,
        }

        # Monday=0 ... Sunday=6
        self.day_of_week_profile = {
            0: 1.20,  # Monday spike
            1: 1.05,
            2: 1.00,
            3: 0.98,
            4: 0.95,
            5: 0.55,
            6: 0.45,
        }

    def _is_holiday(self, ts: pd.Timestamp) -> bool:
        return ts.normalize() in self.us_holidays

    def _holiday_name(self, ts: pd.Timestamp) -> Optional[str]:
        day = ts.normalize()
        if day in self.us_holidays:
            return str(self.us_holidays.get(day))
        return None

    def _seasonality_factor(self, ts: pd.Timestamp, seasonality_type: str) -> float:
        month = ts.month
        if seasonality_type == "retail":
            # Higher demand around holidays / year-end shopping
            if month in (11, 12):
                return 1.35
            if month in (1, 7):
                return 1.10
            return 1.0
        if seasonality_type == "billing":
            # Month-start billing cycles + tax season
            day = ts.day
            factor = 1.25 if day <= 5 else (1.10 if day >= 28 else 1.0)
            if month in (3, 4):
                factor *= 1.20
            return factor
        if seasonality_type == "tech":
            # Product launches / back-to-school / new-year device spikes
            if month in (1, 8, 9):
                return 1.25
            if month in (11, 12):
                return 1.15
            return 1.0
        if seasonality_type == "retention":
            # Competitive switch seasons
            if month in (1, 6, 7):
                return 1.20
            return 1.0
        return 1.0

    def _holiday_volume_factor(self, ts: pd.Timestamp, channel: str) -> float:
        """Holidays suppress live channels; email backlog often rises next day."""
        if not self._is_holiday(ts):
            # Day-after-holiday catch-up
            yesterday = (ts.normalize() - timedelta(days=1))
            if yesterday in self.us_holidays:
                return 1.25 if channel in ("voice", "chat") else 1.15
            return 1.0
        if channel == "voice":
            return 0.35
        if channel == "chat":
            return 0.50
        if channel == "social":
            return 0.80
        return 0.90  # email

    def _trend_factor(self, ts: pd.Timestamp) -> float:
        """Mild growth in offered volume over the generation window."""
        total_days = max((self.end_date - self.start_date).days, 1)
        elapsed = (ts.normalize() - self.start_date).days
        return 1.0 + 0.15 * (elapsed / total_days)

    def _sample_volume(
        self,
        base_volume: float,
        hour_factor: float,
        dow_factor: float,
        season_factor: float,
        holiday_factor: float,
        channel_share: float,
        weekend_factor: float,
        trend_factor: float,
        is_weekend: bool,
    ) -> int:
        weekend_adj = weekend_factor if is_weekend else 1.0
        expected = (
            base_volume
            * hour_factor
            * dow_factor
            * season_factor
            * holiday_factor
            * channel_share
            * weekend_adj
            * trend_factor
        )
        # Over-dispersed count noise (approx. negative-binomial via gamma-poisson)
        if expected <= 0:
            return 0
        lam = self.rng.gamma(shape=max(expected / 2.0, 1e-3), scale=2.0)
        return int(max(0, self.rng.poisson(lam)))

    def _sample_aht_seconds(
        self,
        base_aht: float,
        aht_multiplier: float,
        hour: int,
        volume: int,
        is_holiday: bool,
    ) -> float:
        # Longer handle times at open/close and on sparse/holiday intervals
        open_close = 1.15 if hour in (8, 9, 17, 18) else 1.0
        sparse = 1.10 if volume < 5 else 1.0
        holiday_adj = 1.08 if is_holiday else 1.0
        noise = float(self.rng.normal(1.0, 0.08))
        aht = base_aht * aht_multiplier * open_close * sparse * holiday_adj * noise
        return float(max(30.0, round(aht, 1)))

    def generate_demand_frame(self) -> pd.DataFrame:
        """Return a single DataFrame of hourly contact-center demand records."""
        timestamps = pd.date_range(
            self.start_date,
            self.end_date + pd.Timedelta(hours=23),
            freq="h",
        )
        records: List[Dict] = []

        for ts in timestamps:
            hour = int(ts.hour)
            dow = int(ts.dayofweek)
            is_weekend = dow >= 5
            is_holiday = self._is_holiday(ts)
            holiday_name = self._holiday_name(ts)
            hour_factor = self.hour_of_day_profile[hour]
            dow_factor = self.day_of_week_profile[dow]
            trend = self._trend_factor(ts)

            for bu_name, bu_cfg in self.business_units.items():
                season = self._seasonality_factor(ts, bu_cfg["seasonality"])
                for channel, ch_cfg in self.channels.items():
                    holiday_factor = self._holiday_volume_factor(ts, channel)
                    volume = self._sample_volume(
                        base_volume=bu_cfg["base_volume"],
                        hour_factor=hour_factor,
                        dow_factor=dow_factor,
                        season_factor=season,
                        holiday_factor=holiday_factor,
                        channel_share=ch_cfg["volume_share"],
                        weekend_factor=ch_cfg["weekend_factor"],
                        trend_factor=trend,
                        is_weekend=is_weekend,
                    )
                    aht = self._sample_aht_seconds(
                        base_aht=bu_cfg["base_aht_seconds"],
                        aht_multiplier=ch_cfg["aht_multiplier"],
                        hour=hour,
                        volume=volume,
                        is_holiday=is_holiday,
                    )
                    records.append(
                        {
                            "timestamp": ts,
                            "business_unit": bu_name,
                            "channel": channel,
                            "volume": volume,
                            "average_handle_time": aht,
                            "day_of_week": dow,
                            "day_of_week_name": ts.day_name(),
                            "hour_of_day": hour,
                            "is_holiday": is_holiday,
                            "holiday_name": holiday_name,
                            "is_weekend": is_weekend,
                            "date": ts.normalize(),
                        }
                    )

        df = pd.DataFrame.from_records(records)
        # Stable column order for consumers
        column_order = [
            "timestamp",
            "business_unit",
            "channel",
            "volume",
            "average_handle_time",
            "day_of_week",
            "day_of_week_name",
            "hour_of_day",
            "is_holiday",
            "holiday_name",
            "is_weekend",
            "date",
        ]
        return df[column_order]

    def generate_contact_center_data(
        self, output_dir: str = "/tmp/contact_center_data"
    ) -> Dict[str, List[str]]:
        """
        Generate and persist contact-center demand data.

        Writes daily parquet partitions under contact_center/ plus a combined
        file and generation metadata. Sales generation is untouched.
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info(
            f"Generating contact-center demand data from "
            f"{self.start_date.date()} to {self.end_date.date()}"
        )

        df = self.generate_demand_frame()
        file_paths: Dict[str, List[str]] = {
            "contact_center": [],
            "contact_center_daily": [],
        }

        # Combined dataset for convenient loading
        combined_path = os.path.join(
            output_dir, "contact_center/contact_center_demand.parquet"
        )
        os.makedirs(os.path.dirname(combined_path), exist_ok=True)
        df.to_parquet(combined_path, index=False)
        file_paths["contact_center"].append(combined_path)

        # Daily partitions (mirror sales-style layout)
        for day, day_df in df.groupby("date", sort=True):
            day_ts = pd.Timestamp(day)
            date_str = day_ts.strftime("%Y-%m-%d")
            daily_path = os.path.join(
                output_dir,
                f"contact_center_daily/year={day_ts.year}/"
                f"month={day_ts.month:02d}/day={day_ts.day:02d}/"
                f"demand_{date_str}.parquet",
            )
            os.makedirs(os.path.dirname(daily_path), exist_ok=True)
            day_df.to_parquet(daily_path, index=False)
            file_paths["contact_center_daily"].append(daily_path)

        metadata = {
            "generation_date": datetime.now().isoformat(),
            "dataset_type": "contact_center",
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "n_business_units": len(self.business_units),
            "n_channels": len(self.channels),
            "n_rows": int(len(df)),
            "file_counts": {k: len(v) for k, v in file_paths.items()},
            "total_files": sum(len(v) for v in file_paths.values()),
            "columns": list(df.columns),
        }
        metadata_path = os.path.join(
            output_dir, "metadata/contact_center_generation_metadata.parquet"
        )
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        pd.DataFrame([metadata]).to_parquet(metadata_path, index=False)

        logger.info(
            f"Generated contact-center dataset with {len(df)} rows "
            f"across {metadata['total_files']} files in {output_dir}"
        )
        return file_paths


def generate_synthetic_data(
    dataset: str = "sales",
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    sales_output_dir: str = "/tmp/sales_data",
    contact_center_output_dir: str = "/tmp/contact_center_data",
    seed: Optional[int] = 42,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Optionally generate sales and/or contact-center synthetic datasets.

    Args:
        dataset: One of ``"sales"``, ``"contact_center"``, or ``"both"``.
        start_date / end_date: Inclusive generation window.
        sales_output_dir: Output root for the existing sales generator.
        contact_center_output_dir: Output root for contact-center demand.
        seed: RNG seed for contact-center generation.

    Returns:
        Mapping of dataset name -> file_paths dict from the respective generator.
        Existing sales behavior is unchanged when ``dataset`` is ``"sales"``.
    """
    dataset = dataset.lower().strip()
    allowed = {"sales", "contact_center", "both"}
    if dataset not in allowed:
        raise ValueError(f"dataset must be one of {sorted(allowed)}, got {dataset!r}")

    results: Dict[str, Dict[str, List[str]]] = {}

    if dataset in ("sales", "both"):
        sales_gen = RealisticSalesDataGenerator(
            start_date=start_date, end_date=end_date
        )
        results["sales"] = sales_gen.generate_sales_data(output_dir=sales_output_dir)

    if dataset in ("contact_center", "both"):
        cc_gen = ContactCenterDemandGenerator(
            start_date=start_date, end_date=end_date, seed=seed
        )
        results["contact_center"] = cc_gen.generate_contact_center_data(
            output_dir=contact_center_output_dir
        )

    return results
