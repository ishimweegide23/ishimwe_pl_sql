# ⚽ Premier League Database Management System

[![Oracle](https://img.shields.io/badge/Oracle-PL%2FSQL-red?style=for-the-badge&logo=oracle&logoColor=white)](https://www.oracle.com/)
[![Database](https://img.shields.io/badge/Database-Relational-blue?style=for-the-badge&logo=database&logoColor=white)](https://github.com/ishimweegide23/ishimwe_pl_sql)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

## 📖 Project Overview

This project implements a comprehensive **Football Database Management System** using Oracle PL/SQL. The database is designed to manage Premier League teams, players, matches, statistics, and standings with proper relational structure and data integrity.

## 🏗️ Database Architecture

### 📊 Entity Relationship Diagram b
```
Teams (1) ←→ (N) Players
Teams (1) ←→ (N) Matches (Home/Away)
Players (1) ←→ (N) PlayerStats
Teams (1) ←→ (1) Standings
```

## 🗄️ Database Schema

### 🏆 Teams Table
- **TeamID** (Primary Key)
- **TeamName** - Club name
- **Stadium** - Home stadium
- **Coach/ManagerName** - Current manager
- **FoundedYear** - Year the club was founded

### 👥 Players Table
- **PlayerID** (Primary Key)
- **PlayerName** - Player's full name
- **Position** - Playing position
- **Nationality** - Player's nationality
- **DateOfBirth** - Birth date
- **TeamID** (Foreign Key → Teams)

### ⚽ Matches Table
- **MatchID** (Primary Key)
- **HomeTeamID** (Foreign Key → Teams)
- **AwayTeamID** (Foreign Key → Teams)
- **MatchDate** - Date of the match
- **HomeTeamGoals** - Goals scored by home team
- **AwayTeamGoals** - Goals scored by away team

### 📈 PlayerStats Table
- **StatID** (Primary Key)
- **PlayerID** (Foreign Key → Players)
- **MatchesPlayed** - Total matches played
- **GoalsScored** - Total goals scored
- **Assists** - Total assists

### 🏅 Standings Table
- **TeamID** (Foreign Key → Teams)
- **Points** - Total points earned
- **Wins** - Number of wins
- **Losses** - Number of losses
- **Draws** - Number of draws
- **GoalDifference** - Goal difference

## 🚀 Features Implemented

### ✅ Core Functionality
- [x] **Table Creation** with proper constraints
- [x] **Foreign Key Relationships** for data integrity
- [x] **Data Insertion** with sample Premier League data
- [x] **Table Alterations** (adding columns dynamically)
- [x] **Data Updates** and modifications
- [x] **Data Deletion** operations
- [x] **Inner Joins** for related data retrieval
- [x] **Left Joins** for comprehensive data views
- [x] **Transaction Management** (COMMIT/ROLLBACK)

### 🔍 Query Examples
1. **Player-Team Relationships**: Display players with their respective teams
2. **Team Rosters**: Show all teams with their players (including teams without players)
3. **Match Results**: Track game outcomes and statistics
4. **League Standings**: Monitor team performance and rankings

## 📋 Sample Data

The database includes sample data for popular Premier League teams:
- ⚪ **Liverpool** (Anfield) - Arne Slot
- 🔵 **Manchester City** (Etihad Stadium) - Pep Guardiola
- 🔵 **Chelsea** (Stamford Bridge) - Enzo Maresca
- 🔴 **Arsenal** (Emirates Stadium) - Mikel Arteta

## 🛠️ Installation & Setup

### Prerequisites
- Oracle Database 11g or higher
- SQL Developer or similar Oracle client
- Basic understanding of PL/SQL

### Setup Instructions
1. **Clone the repository**
   ```bash
   git clone https://github.com/ishimweegide23/ishimwe_pl_sql.git
   cd ishimwe_pl_sql
   ```

2. **Connect to Oracle Database**
   ```sql
   -- Connect to your Oracle instance
   sqlplus username/password@database
   ```

3. **Execute the SQL Script**
   ```sql
   @PLSQL_WORK.sql
   ```

## 💻 Usage Examples

### Basic Queries
```sql
-- View all teams and their stadiums
SELECT TeamName, Stadium, ManagerName 
FROM Teams;

-- Get player information with team details
SELECT p.PlayerName, p.Position, t.TeamName
FROM Players p
INNER JOIN Teams t ON p.TeamID = t.TeamID;

-- Check current league standings
SELECT t.TeamName, s.Points, s.Wins, s.Draws, s.Losses
FROM Teams t
INNER JOIN Standings s ON t.TeamID = s.TeamID
ORDER BY s.Points DESC;
```

## 🔧 Advanced Operations

### Transaction Management
```sql
-- Safe data modification
BEGIN
    UPDATE Teams SET ManagerName = 'New Manager' WHERE TeamID = 1;
    -- Verify changes
    SELECT * FROM Teams WHERE TeamID = 1;
    -- Commit if satisfied
    COMMIT;
    -- Or rollback if needed
    -- ROLLBACK;
END;
```

## 📊 Database Statistics

- **4 Core Tables** with proper relationships
- **Sample Data** for 4 Premier League teams
- **Foreign Key Constraints** ensuring data integrity
- **Flexible Schema** supporting future expansions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 Future Enhancements

- [ ] Add Seasons table for historical data
- [ ] Implement Views for complex queries
- [ ] Add Stored Procedures for common operations
- [ ] Create Triggers for automatic data updates
- [ ] Add Indexes for performance optimization
- [ ] Implement User roles and permissions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ishimwe Egide**
- GitHub: [@ishimweegide23](https://github.com/ishimweegide23)
- Project: [PL/SQL Football Database](https://github.com/ishimweegide23/ishimwe_pl_sql)

## 🙏 Acknowledgments

- Oracle Corporation for PL/SQL documentation
- Premier League for inspiration
- Football community for the passion that drives this project

---

⭐ **Star this repository if you found it helpful!** ⭐
