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

-- =====================================================
-- ADD THESE TABLES TO YOUR EXISTING DATABASE SCHEMA
-- =====================================================

-- Users table for authentication
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('admin', 'manager', 'viewer')),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended')),
    last_login TIMESTAMP,
    last_active TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(id),
    profile_image VARCHAR(500),
    phone_number VARCHAR(20),
    zone_id INTEGER REFERENCES zones(id) -- For managers assigned to specific zones
);

-- Login attempts for security
CREATE TABLE login_attempts (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    ip_address INET,
    attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN DEFAULT FALSE
);

-- Session management
CREATE TABLE user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT TRUE
);

-- Password reset tokens
CREATE TABLE password_resets (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    used BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit log for all actions
CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    action VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50), -- 'pothole', 'user', 'alert', etc.
    entity_id INTEGER,
    old_data JSONB,
    new_data JSONB,
    ip_address INET,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Notifications table
CREATE TABLE notifications (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(200) NOT NULL,
    message TEXT,
    type VARCHAR(50) CHECK (type IN ('info', 'success', 'warning', 'danger')),
    related_entity_type VARCHAR(50),
    related_entity_id INTEGER,
    is_read BOOLEAN DEFAULT FALSE,
    read_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Maintenance crews/teams
CREATE TABLE maintenance_crews (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    supervisor_id INTEGER REFERENCES users(id),
    zone_id INTEGER REFERENCES zones(id),
    contact_number VARCHAR(20),
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Crew members
CREATE TABLE crew_members (
    id SERIAL PRIMARY KEY,
    crew_id INTEGER REFERENCES maintenance_crews(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id),
    role VARCHAR(50) CHECK (role IN ('supervisor', 'technician', 'driver', 'laborer')),
    joined_date DATE DEFAULT CURRENT_DATE,
    active BOOLEAN DEFAULT TRUE
);

-- Repair jobs (track repairs)
CREATE TABLE repair_jobs (
    id SERIAL PRIMARY KEY,
    pothole_id INTEGER REFERENCES potholes(id) ON DELETE CASCADE,
    crew_id INTEGER REFERENCES maintenance_crews(id),
    assigned_by INTEGER REFERENCES users(id),
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'cancelled')),
    notes TEXT,
    materials_used JSONB, -- Store materials used as JSON
    before_photos JSONB, -- Array of photo paths
    after_photos JSONB, -- Array of photo paths
    quality_check_passed BOOLEAN,
    quality_check_by INTEGER REFERENCES users(id),
    quality_check_at TIMESTAMP
);

-- Materials inventory
CREATE TABLE materials (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    unit VARCHAR(20), -- kg, liters, bags, etc.
    quantity FLOAT DEFAULT 0,
    reorder_level FLOAT,
    last_restocked TIMESTAMP
);

-- Repair materials used (junction table)
CREATE TABLE repair_materials (
    id SERIAL PRIMARY KEY,
    repair_job_id INTEGER REFERENCES repair_jobs(id) ON DELETE CASCADE,
    material_id INTEGER REFERENCES materials(id),
    quantity_used FLOAT,
    cost_per_unit DECIMAL(10, 2)
);

-- Reports table (for generated reports)
CREATE TABLE reports (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    type VARCHAR(50) CHECK (type IN ('daily', 'weekly', 'monthly', 'custom', 'zone')),
    generated_by INTEGER REFERENCES users(id),
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_path VARCHAR(500),
    parameters JSONB, -- Store report parameters
    data_summary JSONB -- Store summary data for quick access
);

-- Insert sample potholes (if you don't have any yet)
INSERT INTO potholes (location, latitude, longitude, size_classification, diameter, status, reported_by) VALUES
(ST_SetSRID(ST_MakePoint(31.0335, -17.8252), 4326), -17.8252, 31.0335, 'large', 1.2, 'pending', 'anonymous'),
(ST_SetSRID(ST_MakePoint(31.0400, -17.8300), 4326), -17.8300, 31.0400, 'medium', 0.5, 'verified', 'manager1'),
(ST_SetSRID(ST_MakePoint(31.0250, -17.8200), 4326), -17.8200, 31.0250, 'small', 0.3, 'repaired', 'anonymous');

-- Insert sample alerts
INSERT INTO alerts (type, pothole_id, message) VALUES
('large_pothole', 1, 'Critical pothole detected in Harare CBD requiring immediate attention'),
('high_density', NULL, 'High density area detected in Bulawayo with multiple potholes');


-- Insert sample zones for Zimbabwe
INSERT INTO zones (name, boundary) VALUES
('Harare CBD', ST_GeomFromText('POLYGON((31.0 -17.8, 31.1 -17.8, 31.1 -17.9, 31.0 -17.9, 31.0 -17.8))', 4326)),
('Mbare', ST_GeomFromText('POLYGON((31.1 -17.9, 31.2 -17.9, 31.2 -18.0, 31.1 -18.0, 31.1 -17.9))', 4326)),
('Bulawayo CBD', ST_GeomFromText('POLYGON((28.5 -20.1, 28.6 -20.1, 28.6 -20.2, 28.5 -20.2, 28.5 -20.1))', 4326));