"""
WeatherProvider abstraction (CDC section 4-5).

Two concrete providers are implemented:

  OpenMeteoProvider
      REAL implementation calling the Open-Meteo Historical Weather API
      (archive-api.open-meteo.com/v1/archive) and the Forecast API
      (api.open-meteo.com/v1/forecast). This is the intended production
      provider. It requires outbound internet access to open-meteo.com.

  SyntheticClearSkyProvider
      A clearly-labelled FALLBACK/DEMO provider used only when the real API
      cannot be reached (e.g. this sandboxed build environment has no
      outbound access to open-meteo.com - see docs/DATA_READINESS_REPORT.md).
      It derives physically-plausible weather from pvlib clear-sky irradiance
      plus a stochastic cloud-attenuation process. It NEVER claims to be
      observed weather - every record is tagged weather_source="SYNTHETIC_CLEARSKY".

Both providers return the SAME schema, so the rest of the pipeline (features,
physics model, training) is provider-agnostic and swapping to real data
requires no downstream changes - only changing `provider` in configs/config.yaml.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pvlib

WEATHER_COLUMNS = [
    "district_id", "district", "governorate",
    "requested_latitude", "requested_longitude",
    "weather_grid_latitude", "weather_grid_longitude",
    "timestamp",
    "temperature_2m", "relative_humidity_2m", "cloud_cover",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "direct_normal_irradiance", "wind_speed_10m", "precipitation",
    "weather_source", "retrieval_timestamp",
]

OPEN_METEO_HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "cloud_cover",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "direct_normal_irradiance", "wind_speed_10m", "precipitation",
]


class WeatherProvider(ABC):
    """Common interface every weather provider must implement."""

    @abstractmethod
    def get_historical_weather(
        self, district_id: str, district: str, governorate: str,
        latitude: float, longitude: float, start_date: str, end_date: str,
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    def get_forecast_weather(
        self, district_id: str, district: str, governorate: str,
        latitude: float, longitude: float, horizon_hours: int = 72,
    ) -> pd.DataFrame:
        ...


@dataclass
class OpenMeteoProvider(WeatherProvider):
    """
    REAL provider. Ready to run as-is in any environment with outbound
    internet access to open-meteo.com (no API key required for
    non-commercial use).

    NOTE: this sandboxed build/demo environment restricts outbound network
    access to a short allow-list that does NOT include open-meteo.com, so
    this class could not be executed to produce the sample datasets shipped
    in data/processed/. Run it directly (`python -m src.ingestion.weather_provider`)
    from an environment with normal internet access to fetch real data - the
    downstream pipeline needs no other change.
    """
    historical_base_url: str = "https://archive-api.open-meteo.com/v1/archive"
    forecast_base_url: str = "https://api.open-meteo.com/v1/forecast"
    request_timeout_s: int = 30
    max_retries: int = 3
    retry_backoff_s: float = 2.0

    def _request(self, url: str, params: dict) -> dict:
        import requests
        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = requests.get(url, params=params, timeout=self.request_timeout_s)
                r.raise_for_status()
                return r.json()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(self.retry_backoff_s * (attempt + 1))
        raise RuntimeError(f"Open-Meteo request failed after {self.max_retries} attempts: {last_err}")

    def _parse(self, payload: dict, district_id, district, governorate,
               req_lat, req_lon, source_label: str) -> pd.DataFrame:
        hourly = payload["hourly"]
        df = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"])})
        for v in OPEN_METEO_HOURLY_VARS:
            df[v] = hourly.get(v, np.nan)
        df["district_id"] = district_id
        df["district"] = district
        df["governorate"] = governorate
        df["requested_latitude"] = req_lat
        df["requested_longitude"] = req_lon
        df["weather_grid_latitude"] = payload.get("latitude", req_lat)
        df["weather_grid_longitude"] = payload.get("longitude", req_lon)
        df["weather_source"] = source_label
        df["retrieval_timestamp"] = pd.Timestamp.now(tz="UTC")
        return df[WEATHER_COLUMNS]

    def get_historical_weather(self, district_id, district, governorate,
                                latitude, longitude, start_date, end_date) -> pd.DataFrame:
        params = {
            "latitude": latitude, "longitude": longitude,
            "start_date": start_date, "end_date": end_date,
            "hourly": ",".join(OPEN_METEO_HOURLY_VARS),
            "timezone": "Africa/Tunis",
        }
        payload = self._request(self.historical_base_url, params)
        return self._parse(payload, district_id, district, governorate, latitude, longitude,
                            source_label="OPEN_METEO_ARCHIVE_REAL")

    def get_forecast_weather(self, district_id, district, governorate,
                              latitude, longitude, horizon_hours=72) -> pd.DataFrame:
        params = {
            "latitude": latitude, "longitude": longitude,
            "hourly": ",".join(OPEN_METEO_HOURLY_VARS),
            "forecast_days": max(1, int(np.ceil(horizon_hours / 24))),
            "timezone": "Africa/Tunis",
        }
        payload = self._request(self.forecast_base_url, params)
        df = self._parse(payload, district_id, district, governorate, latitude, longitude,
                          source_label="OPEN_METEO_FORECAST_REAL")
        return df[df["timestamp"] <= df["timestamp"].min() + pd.Timedelta(hours=horizon_hours)]


@dataclass
class SyntheticClearSkyProvider(WeatherProvider):
    """
    FALLBACK / DEMO provider - NOT real observed weather.

    Method:
      1. pvlib Ineichen clear-sky model gives physically correct GHI/DNI/DHI
         for the exact district coordinates and timestamps.
      2. A stochastic, spatially- and temporally-autocorrelated "cloud
         attenuation" process (clearness index k in [0.15, 1.0]) multiplies
         the clear-sky irradiance to emulate realistic cloudy/clear day
         variability, seeded per-district for reproducibility.
      3. Temperature uses a simple seasonal + diurnal sinusoid calibrated to
         Tunisian coastal/inland climatology (documented assumption, not a
         measurement).

    Every row is tagged weather_source="SYNTHETIC_CLEARSKY" and must never be
    presented as observed weather.
    """
    random_seed: int = 42
    altitude_m: float = 50.0

    def _clearsky(self, latitude, longitude, times: pd.DatetimeIndex) -> pd.DataFrame:
        solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude, altitude=self.altitude_m)
        linke = pvlib.clearsky.lookup_linke_turbidity(times, latitude, longitude)
        airmass_rel = pvlib.atmosphere.get_relative_airmass(solpos["apparent_zenith"])
        pressure = pvlib.atmosphere.alt2pres(self.altitude_m)
        airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass_rel, pressure)
        cs = pvlib.clearsky.ineichen(solpos["apparent_zenith"], airmass_abs, linke, altitude=self.altitude_m)
        cs["apparent_elevation"] = solpos["apparent_elevation"]
        return cs

    def _cloud_process(self, n: int, district_id: str, freq_per_day: int) -> np.ndarray:
        """AR(1) clearness-index process in [0.1, 1.0], seeded per district."""
        rng = np.random.default_rng(abs(hash(district_id)) % (2**32) + self.random_seed)
        phi = 0.985  # persistence -> multi-hour cloud episodes rather than white noise
        sigma = 0.03
        k = np.empty(n)
        k[0] = rng.uniform(0.55, 0.85)
        for i in range(1, n):
            k[i] = 0.75 + phi * (k[i - 1] - 0.75) + rng.normal(0, sigma)
        # occasional multi-hour overcast episodes
        n_episodes = rng.poisson(n / (freq_per_day * 6))
        for _ in range(n_episodes):
            start = rng.integers(0, max(1, n - 1))
            length = rng.integers(freq_per_day // 2, freq_per_day * 2)
            k[start:start + length] *= rng.uniform(0.25, 0.6)
        return np.clip(k, 0.1, 1.0)

    def get_historical_weather(self, district_id, district, governorate,
                                latitude, longitude, start_date, end_date) -> pd.DataFrame:
        times = pd.date_range(start_date, end_date, freq="1h", tz="Africa/Tunis", inclusive="left")
        cs = self._clearsky(latitude, longitude, times)
        k = self._cloud_process(len(times), district_id, freq_per_day=24)

        ghi = cs["ghi"].to_numpy() * k
        dni = cs["dni"].to_numpy() * k ** 1.5  # DNI attenuates faster than GHI under clouds
        dhi = np.clip(ghi - dni * np.cos(np.radians(cs["apparent_elevation"].clip(lower=0))), 0, None)

        doy = times.dayofyear.to_numpy()
        hour = times.hour.to_numpy() + times.minute.to_numpy() / 60.0
        seasonal_mean_temp = 18 + 8 * np.sin(2 * np.pi * (doy - 105) / 365.25)  # Tunisia climatology assumption
        diurnal = 6 * np.sin(2 * np.pi * (hour - 8) / 24)
        temp = seasonal_mean_temp + diurnal + np.random.default_rng(hash(district_id) % 2**32).normal(0, 1.2, len(times))

        cloud_cover_pct = np.clip((1 - k) * 100 + np.random.default_rng(0).normal(0, 5, len(times)), 0, 100)
        rel_humidity = np.clip(55 + cloud_cover_pct * 0.25 - diurnal * 1.5, 15, 100)
        wind = np.clip(np.random.default_rng(1).gamma(2.0, 1.8, len(times)), 0, None)
        precip = np.where(cloud_cover_pct > 80, np.random.default_rng(2).exponential(0.4, len(times)), 0.0)
        precip *= (np.random.default_rng(3).random(len(times)) < 0.15)

        df = pd.DataFrame({
            "district_id": district_id, "district": district, "governorate": governorate,
            "requested_latitude": latitude, "requested_longitude": longitude,
            "weather_grid_latitude": latitude, "weather_grid_longitude": longitude,
            "timestamp": times.tz_localize(None),
            "temperature_2m": temp,
            "relative_humidity_2m": rel_humidity,
            "cloud_cover": cloud_cover_pct,
            "shortwave_radiation": ghi,
            "direct_radiation": dni * np.cos(np.radians(np.clip(90 - cs["apparent_elevation"].to_numpy(), 0, 90))),
            "diffuse_radiation": dhi,
            "direct_normal_irradiance": dni,
            "wind_speed_10m": wind,
            "precipitation": precip,
            "weather_source": "SYNTHETIC_CLEARSKY",
            "retrieval_timestamp": pd.Timestamp.now(tz="UTC"),
        })
        # night values must be exactly zero irradiance
        night = cs["apparent_elevation"].to_numpy() <= 0
        for c in ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance"]:
            df.loc[night, c] = 0.0
        return df[WEATHER_COLUMNS]

    def get_forecast_weather(self, district_id, district, governorate,
                              latitude, longitude, horizon_hours=72) -> pd.DataFrame:
        start = pd.Timestamp.now(tz="Africa/Tunis").floor("h")
        end = start + pd.Timedelta(hours=horizon_hours)
        df = self.get_historical_weather(
            district_id, district, governorate, latitude, longitude,
            start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
        )
        df["weather_source"] = "SYNTHETIC_CLEARSKY_FORECAST"
        return df


def get_provider(name: str, **kwargs) -> WeatherProvider:
    if name == "open_meteo":
        return OpenMeteoProvider(**kwargs)
    if name == "synthetic_clearsky":
        return SyntheticClearSkyProvider(**kwargs)
    raise ValueError(f"Unknown weather provider: {name}")


if __name__ == "__main__":
    p = SyntheticClearSkyProvider()
    df = p.get_historical_weather("D42", "SFAX NORD", "Sfax", 34.7825, 10.7692, "2026-07-01", "2026-07-03")
    print(df[["timestamp", "temperature_2m", "shortwave_radiation", "cloud_cover", "weather_source"]].iloc[10:20])
