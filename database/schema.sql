-- Enable PostGIS for spatial queries
CREATE EXTENSION IF NOT EXISTS postgis;

-- Zones table
CREATE TABLE zones (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    boundary GEOMETRY(POLYGON, 4326),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Potholes table
CREATE TABLE potholes (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    image_path VARCHAR(500),
    size_classification VARCHAR(20),
    diameter FLOAT,
    confidence_score FLOAT,
    status VARCHAR(20) DEFAULT 'pending',
    reported_by VARCHAR(100),
    reported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    verified_at TIMESTAMP,
    repaired_at TIMESTAMP,
    zone_id INTEGER REFERENCES zones(id),

    -- Create spatial index
    CONSTRAINT valid_status CHECK (status IN ('pending', 'verified', 'repaired'))
);

CREATE INDEX idx_potholes_location ON potholes USING GIST (location);
CREATE INDEX idx_potholes_status ON potholes(status);

-- Alerts table
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    type VARCHAR(50) NOT NULL,
    pothole_id INTEGER REFERENCES potholes(id),
    zone_id INTEGER REFERENCES zones(id),
    message TEXT,
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged BOOLEAN DEFAULT FALSE,

    CONSTRAINT valid_alert_type CHECK (type IN ('large_pothole', 'high_density'))
);

-- Insert sample zones for Zimbabwe
INSERT INTO zones (name, boundary) VALUES
('Harare CBD', ST_GeomFromText('POLYGON((31.0 -17.8, 31.1 -17.8, 31.1 -17.9, 31.0 -17.9, 31.0 -17.8))', 4326)),
('Mbare', ST_GeomFromText('POLYGON((31.1 -17.9, 31.2 -17.9, 31.2 -18.0, 31.1 -18.0, 31.1 -17.9))', 4326)),
('Bulawayo CBD', ST_GeomFromText('POLYGON((28.5 -20.1, 28.6 -20.1, 28.6 -20.2, 28.5 -20.2, 28.5 -20.1))', 4326));