from models import init_db, User

def create_admin():
    # Initialize the SQLite database and table if not exists
    init_db()

    admin_name = "Administrator"
    admin_mobile = "9999999999"
    admin_email = "admin@eeg.com"
    admin_address = "Head Office"
    admin_password = "admin"
    admin_role = "admin"

    print("=== Admin User Setup ===")

    # Check if admin already exists
    existing_admin = User.get_by_email(admin_email)
    if existing_admin:
        print(f"ℹ️ Admin user with email '{admin_email}' already exists.")
        return

    # Create the admin user
    success = User.create(
        name=admin_name,
        mobile=admin_mobile,
        email=admin_email,
        address=admin_address,
        password=admin_password,
        role=admin_role
    )

    if success:
        print(f"✅ Default admin created successfully!")
        print(f"📧 Email: {admin_email}")
        print(f"🔑 Password: {admin_password}")
    else:
        print(f"❌ Error: Could not create admin user (email may already exist).")

if __name__ == '__main__':
    create_admin()
