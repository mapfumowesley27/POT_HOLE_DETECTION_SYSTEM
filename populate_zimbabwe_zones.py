from __future__ import annotations

import json
import os
import random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from werkzeug.security import generate_password_hash

# Add backend to path so imports work regardless of current working directory
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import create_app, db
from app.models.maintenance import CrewMember, MaintenanceCrew, Material, RepairJob, RepairMaterial
from app.models.pothole import Alert, Pothole, Zone
from app.models.user import User

RNG = random.Random(42)
DEFAULT_PASSWORD_ENV = "SEED_DEFAULT_PASSWORD"
DESTRUCTIVE_FLAG = "--force-reset"


def get_zimbabwe_zones() -> list[dict[str, object]]:
    return [
        {"id": 1, "name": "Harare City Council"},
        {"id": 2, "name": "Bulawayo City Council"},
        {"id": 3, "name": "Chitungwiza Municipality"},
        {"id": 4, "name": "Mutare City Council"},
        {"id": 5, "name": "Gweru City Council"},
        {"id": 6, "name": "Kwekwe City Council"},
        {"id": 7, "name": "Kadoma City Council"},
        {"id": 8, "name": "Masvingo Municipality"},
        {"id": 9, "name": "Chinhoyi Municipality"},
        {"id": 10, "name": "Marondera Municipality"},
        {"id": 11, "name": "Redcliff Municipality"},
        {"id": 12, "name": "Norton Town Council"},
        {"id": 13, "name": "Ruwa Local Board"},
        {"id": 14, "name": "Epworth Local Board"},
        {"id": 15, "name": "Victoria Falls Municipality"},
        {"id": 16, "name": "Hwange Town Council"},
        {"id": 17, "name": "Beitbridge Town Council"},
        {"id": 18, "name": "Plumtree Town Council"},
        {"id": 19, "name": "Rusape Town Council"},
        {"id": 20, "name": "Chipinge Town Council"},
        {"id": 21, "name": "Goromonzi RDC"},
        {"id": 22, "name": "Murehwa RDC"},
        {"id": 23, "name": "Mutoko RDC"},
        {"id": 24, "name": "Chikomba RDC"},
        {"id": 25, "name": "Wedza RDC"},
        {"id": 26, "name": "Hwedza RDC"},
        {"id": 27, "name": "Buhera RDC"},
        {"id": 28, "name": "Makoni RDC"},
        {"id": 29, "name": "Nyanga RDC"},
        {"id": 30, "name": "Chimanimani RDC"},
        {"id": 31, "name": "Chipinge RDC"},
        {"id": 32, "name": "Bikita RDC"},
        {"id": 33, "name": "Zaka RDC"},
        {"id": 34, "name": "Masvingo RDC"},
        {"id": 35, "name": "Chivi RDC"},
        {"id": 36, "name": "Mberengwa RDC"},
        {"id": 37, "name": "Zvishavane RDC"},
        {"id": 38, "name": "Gokwe North RDC"},
        {"id": 39, "name": "Gokwe South RDC"},
        {"id": 40, "name": "Binga RDC"},
        {"id": 41, "name": "Lupane RDC"},
        {"id": 42, "name": "Hwange RDC"},
        {"id": 43, "name": "Tsholotsho RDC"},
        {"id": 44, "name": "Umguza RDC"},
        {"id": 45, "name": "Bubi RDC"},
    ]


def get_seed_password() -> str:
    password = os.environ.get(DEFAULT_PASSWORD_ENV)
    if not password:
        raise RuntimeError(
            f"Missing required environment variable: {DEFAULT_PASSWORD_ENV}. "
            f"Set it before running the population script."
        )
    return password


def hash_pass(password: str) -> str:
    return generate_password_hash(password)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def make_zone_identity(zone_id: int, zone_name: str) -> str:
    return f"{zone_id:02d}_{slugify(zone_name)}"


def make_mock_boundary(index: int) -> str:
    row = index // 5
    col = index % 5

    base_lon = 28.0 + (col * 0.8)
    base_lat = -20.5 + (row * 0.5)
    width = 0.18
    height = 0.12

    point_1 = [round(base_lon, 6), round(base_lat, 6)]
    point_2 = [round(base_lon + width, 6), round(base_lat, 6)]
    point_3 = [round(base_lon + width, 6), round(base_lat - height, 6)]
    point_4 = [round(base_lon, 6), round(base_lat - height, 6)]

    boundary = {
        "type": "Polygon",
        "coordinates": [[point_1, point_2, point_3, point_4, point_1]],
    }
    return json.dumps(boundary)


def random_phone(prefix: str) -> str:
    return f"{prefix}{RNG.randint(1000000, 9999999)}"


def reset_database() -> None:
    print("Clearing database tables...")
    db.session.query(RepairMaterial).delete()
    db.session.query(RepairJob).delete()
    db.session.query(CrewMember).delete()
    db.session.query(MaintenanceCrew).delete()
    db.session.query(Material).delete()
    db.session.query(Alert).delete()
    db.session.query(Pothole).delete()
    db.session.query(User).delete()
    db.session.query(Zone).delete()


def seed_zones() -> list[Zone]:
    print("Populating 45 Zimbabwe Zones...")
    zones: list[Zone] = []

    for index, zone_data in enumerate(get_zimbabwe_zones()):
        zone = Zone(
            id=zone_data["id"],
            name=zone_data["name"],
            boundary=make_mock_boundary(index),
        )
        db.session.add(zone)
        zones.append(zone)

    db.session.flush()
    return zones


def seed_users(zones: list[Zone], password_hash: str) -> tuple[list[User], list[User], list[User]]:
    print("Creating users for all zones...")
    admins: list[User] = []
    managers: list[User] = []
    reporters: list[User] = []

    super_admin = User(
        username="super_admin",
        email="superadmin@pothole.gov.zw",
        password_hash=password_hash,
        role="admin",
        full_name="National Super Admin",
        phone_number=random_phone("+26377"),
        status="active",
    )
    db.session.add(super_admin)

    for zone in zones:
        identity = make_zone_identity(zone.id, zone.name)

        admin = User(
            username=f"{identity}_admin",
            email=f"admin.{identity}@pothole.local",
            password_hash=password_hash,
            role="admin",
            full_name=f"{zone.name} Administrator",
            phone_number=random_phone("+26377"),
            zone_id=zone.id,
            status="active",
        )
        db.session.add(admin)
        admins.append(admin)

        manager = User(
            username=f"{identity}_manager",
            email=f"manager.{identity}@pothole.local",
            password_hash=password_hash,
            role="manager",
            full_name=f"{zone.name} Maintenance Manager",
            phone_number=random_phone("+26378"),
            zone_id=zone.id,
            status="active",
        )
        db.session.add(manager)
        managers.append(manager)

        for reporter_index in range(1, 3):
            reporter = User(
                username=f"{identity}_reporter{reporter_index}",
                email=f"reporter{reporter_index}.{identity}@pothole.local",
                password_hash=password_hash,
                role="reporter",
                full_name=f"{zone.name} Reporter {reporter_index}",
                phone_number=random_phone("+26371"),
                zone_id=zone.id,
                status="active",
            )
            db.session.add(reporter)
            reporters.append(reporter)

    db.session.flush()
    print(f"Created {1 + len(admins) + len(managers) + len(reporters)} users.")
    return admins, managers, reporters


def seed_materials() -> list[Material]:
    print("Populating materials inventory...")
    materials_list = [
        ("Cold Mix Asphalt", "bags", 500, 50, 25.50),
        ("Hot Mix Asphalt", "tonnes", 100, 10, 120.00),
        ("Bitumen Emulsion", "litres", 2000, 200, 2.50),
        ("Crushed Stone", "tonnes", 50, 5, 45.00),
        ("Road Marking Paint", "litres", 200, 20, 15.00),
        ("Sealant", "cartridges", 300, 30, 12.00),
    ]

    db_materials: list[Material] = []
    restocked_at = datetime.utcnow() - timedelta(days=5)

    for name, unit, qty, reorder, cost in materials_list:
        material = Material(
            name=name,
            unit=unit,
            quantity=qty,
            reorder_level=reorder,
            cost_per_unit=cost,
            last_restocked=restocked_at,
        )
        db.session.add(material)
        db_materials.append(material)

    db.session.flush()
    return db_materials


def seed_crews(zones: list[Zone], managers: list[User]) -> list[MaintenanceCrew]:
    print("Creating maintenance crews and members for major cities...")
    crews: list[MaintenanceCrew] = []
    crew_roles = ["supervisor", "driver", "laborer"]

    for index in range(10):
        zone = zones[index]
        manager = managers[index]

        crew = MaintenanceCrew(
            name=f"{zone.name.split()[0]} Alpha Crew",
            supervisor_id=manager.id,
            zone_id=zone.id,
            contact_number=random_phone("+26377"),
            active=True,
        )
        db.session.add(crew)
        db.session.flush()

        for member_index, role in enumerate(crew_roles, start=1):
            member = CrewMember(
                crew_id=crew.id,
                name=f"{zone.name.split()[0]} Crew Member {member_index}",
                role=role,
                phone=random_phone("+26371"),
                active=True,
            )
            db.session.add(member)

        crews.append(crew)

    db.session.flush()
    return crews


def seed_potholes_and_repairs(
    zones: list[Zone],
    reporters: list[User],
    materials: list[Material],
) -> None:
    print("Generating potholes, alerts, and repair jobs...")
    statuses = ["pending", "verified", "repaired"]
    sizes = ["small", "medium", "large"]
    reporter_lookup: dict[int, list[User]] = {}

    for reporter in reporters:
        reporter_lookup.setdefault(reporter.zone_id, []).append(reporter)

    for zone_index, zone in enumerate(zones):
        crew = MaintenanceCrew.query.filter_by(zone_id=zone.id).first()
        zone_reporters = reporter_lookup.get(zone.id, [])

        base_lon = 28.0 + ((zone_index % 5) * 0.8) + 0.05
        base_lat = -20.5 + ((zone_index // 5) * 0.5) - 0.03

        for _ in range(RNG.randint(3, 5)):
            size = RNG.choice(sizes)
            status = RNG.choice(statuses)
            diameter = {"small": 0.3, "medium": 0.7, "large": 1.5}[size]
            reported_at = datetime.utcnow() - timedelta(days=RNG.randint(1, 30))
            verified_at = reported_at + timedelta(hours=6) if status in {"verified", "repaired"} else None
            repaired_at = verified_at + timedelta(days=1, hours=4) if status == "repaired" and verified_at else None

            reporter = RNG.choice(zone_reporters) if zone_reporters else None

            pothole = Pothole(
                latitude=round(base_lat + RNG.uniform(-0.04, 0.04), 6),
                longitude=round(base_lon + RNG.uniform(-0.05, 0.05), 6),
                size_classification=size,
                diameter=diameter,
                confidence_score=round(RNG.uniform(0.80, 0.99), 2),
                status=status,
                reported_by=reporter.username if reporter else "anonymous",
                zone_id=zone.id,
                reported_at=reported_at,
                verified_at=verified_at,
                repaired_at=repaired_at,
            )
            db.session.add(pothole)
            db.session.flush()

            if size == "large":
                alert = Alert(
                    type="large_pothole",
                    pothole_id=pothole.id,
                    zone_id=zone.id,
                    message=f"CRITICAL: Large pothole ({diameter}m) detected in {zone.name}",
                    sent_at=reported_at,
                    acknowledged=False,
                )
                db.session.add(alert)

            if status in {"verified", "repaired"}:
                job = RepairJob(
                    pothole_id=pothole.id,
                    crew_id=crew.id if crew else None,
                    assigned_by=crew.supervisor_id if crew else None,
                    assigned_at=reported_at + timedelta(hours=2),
                    started_at=verified_at + timedelta(hours=2) if status == "repaired" and verified_at else None,
                    completed_at=repaired_at if status == "repaired" else None,
                    status="completed" if status == "repaired" else "pending",
                    notes="Standard asphalt repair" if status == "repaired" else "Awaiting crew dispatch",
                )
                db.session.add(job)
                db.session.flush()

                if status == "repaired":
                    for material in RNG.sample(materials, 2):
                        repair_material = RepairMaterial(
                            repair_job_id=job.id,
                            material_id=material.id,
                            quantity_used=round(RNG.uniform(5, 20), 2),
                            cost_per_unit=material.cost_per_unit,
                        )
                        db.session.add(repair_material)


def populate(force_reset: bool = False) -> None:
    if not force_reset:
        raise RuntimeError(
            f"This script is destructive. Re-run with {DESTRUCTIVE_FLAG} to continue."
        )

    password = get_seed_password()
    password_hash = hash_pass(password)

    app = create_app()
    with app.app_context():
        print("--- SYSTEM RE-INITIALIZATION ---")
        try:
            reset_database()
            zones = seed_zones()
            _, managers, reporters = seed_users(zones, password_hash)
            materials = seed_materials()
            seed_crews(zones, managers)
            seed_potholes_and_repairs(zones, reporters, materials)

            db.session.commit()
            print("Population complete. Seed data created successfully.")
        except Exception:
            db.session.rollback()
            raise


if __name__ == "__main__":
    populate(force_reset=DESTRUCTIVE_FLAG in sys.argv)