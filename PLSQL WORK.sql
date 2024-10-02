-- 1. Creating the Teams Table
CREATE TABLE Teams (
    TeamID NUMBER PRIMARY KEY,
    TeamName VARCHAR2(100) NOT NULL,
    Stadium VARCHAR2(100),
    Coach VARCHAR2(100)
);


-- 2. Creating the Players Table
CREATE TABLE Players (
    PlayerID NUMBER PRIMARY KEY,
    PlayerName VARCHAR2(100) NOT NULL,
    Position VARCHAR2(50),
    Nationality VARCHAR2(50),
    DateOfBirth DATE NOT NULL,
    TeamID NUMBER,
    FOREIGN KEY (TeamID) REFERENCES Teams(TeamID)
);



-- 3. Creating the Matches Table
CREATE TABLE Matches (
    MatchID NUMBER PRIMARY KEY,
    HomeTeamID NUMBER,
    AwayTeamID NUMBER,
    MatchDate DATE,
    HomeTeamGoals NUMBER,
    AwayTeamGoals NUMBER,
    FOREIGN KEY (HomeTeamID) REFERENCES Teams(TeamID),
    FOREIGN KEY (AwayTeamID) REFERENCES Teams(TeamID)
);



-- 4. Creating the PlayerStats Table
CREATE TABLE PlayerStats (
    StatID NUMBER PRIMARY KEY,
    PlayerID NUMBER,
    MatchesPlayed NUMBER,
    GoalsScored NUMBER,
    Assists NUMBER,
    FOREIGN KEY (PlayerID) REFERENCES Players(PlayerID)
);


select*from matches;

INSERT INTO Teams (TeamID, TeamName, Stadium, MANAGERNAME)
VALUES (1, 'Liverpool', 'Anfield', 'Jurgen Klopp');

INSERT INTO Teams (TeamID, TeamName, Stadium, MANAGERNAME)
VALUES (2, 'Manchester City', 'Etihad Stadium', 'Pep Guardiola');

INSERT INTO Teams (TeamID, TeamName, Stadium, MANAGERNAME)
VALUES (3, 'Chelsea', 'Stamford Bridge', 'Enzo Marseca');

INSERT INTO Teams (TeamID, TeamName, Stadium, MANAGERNAME)
VALUES (4, 'Arsenal', 'Emirates Stadium', 'Mikel Arteta');

INSERT INTO Players (PlayerID, PlayerName, Position, Nationality, DateOfBirth, TeamID)
VALUES (101, 'Virgil van Dijk', 'Defender', 'Netherlands', TO_DATE('1991-07-08', 'YYYY-MM-DD'), 1);

INSERT INTO Players (PlayerID, PlayerName, Position, Nationality, DateOfBirth, TeamID)
VALUES (102, 'Mohamed Salah', 'Forward', 'Egypt', TO_DATE('1992-06-15', 'YYYY-MM-DD'), 1);

INSERT INTO Players (PlayerID, PlayerName, Position, Nationality, DateOfBirth, TeamID)
VALUES (103, 'Kevin De Bruyne', 'Midfielder', 'Belgium', TO_DATE('1991-06-28', 'YYYY-MM-DD'), 2);

INSERT INTO Players (PlayerID, PlayerName, Position, Nationality, DateOfBirth, TeamID)
VALUES (104, 'Bukayo Saka', 'Midfielder', 'England', TO_DATE('2001-09-05', 'YYYY-MM-DD'), 4);

INSERT INTO Matches (MatchID, HomeTeamID, AwayTeamID, MatchDate, HomeTeamGoals, AwayTeamGoals)
VALUES (1001, 1, 2, TO_DATE('2024-09-15', 'YYYY-MM-DD'), 3, 2);

INSERT INTO Matches (MatchID, HomeTeamID, AwayTeamID, MatchDate, HomeTeamGoals, AwayTeamGoals)
VALUES (1002, 3, 4, TO_DATE('2024-09-16', 'YYYY-MM-DD'), 1, 1);

INSERT INTO PlayerStats (StatID, PlayerID, MatchesPlayed, GoalsScored, Assists)
VALUES (1, 101, 5, 2, 1);

INSERT INTO PlayerStats (StatID, PlayerID, MatchesPlayed, GoalsScored, Assists)
VALUES (2, 102, 5, 4, 2);

INSERT INTO PlayerStats (StatID, PlayerID, MatchesPlayed, GoalsScored, Assists)
VALUES (3, 103, 5, 2, 3);

INSERT INTO PlayerStats (StatID, PlayerID, MatchesPlayed, GoalsScored, Assists)
VALUES (4, 104, 5, 3, 4);

-- Add the FoundedYear column to the existing Teams table
ALTER TABLE Teams
ADD (FoundedYear NUMBER);

UPDATE Teams
SET FoundedYear = 1892
WHERE TeamID = 1;

UPDATE Teams
SET FoundedYear = 1880
WHERE TeamID = 2;

UPDATE Teams
SET FoundedYear = 1905
WHERE TeamID = 3;

UPDATE Teams
SET FoundedYear = 1886
WHERE TeamID = 4;

UPDATE Teams
SET MANAGERNAME = 'Alen Slot'
WHERE TeamName = 'Liverpool';

INSERT INTO Standings (TeamID, Points, Wins, Losses, Draws, GoalDifference)
VALUES (1, 16, 5, 1, 1, 10);

INSERT INTO Standings (TeamID, Points, Wins, Losses, Draws, GoalDifference)
VALUES (2, 18, 6, 1, 0, 12);

INSERT INTO Standings (TeamID, Points, Wins, Losses, Draws, GoalDifference)
VALUES (4, 11, 3, 4, 2, 1);


INSERT INTO Standings (TeamID, Points, Wins, Losses, Draws, GoalDifference)
VALUES (3, 18, 5, 0, 3, 11);

select *from  standings;

DELETE FROM PlayerStats WHERE PlayerID = 101;

SELECT Players.PlayerName, Teams.TeamName
FROM Players
INNER JOIN Teams ON Players.TeamID = Teams.TeamID;

SELECT Teams.TeamName, Players.PlayerName
FROM Teams
LEFT JOIN Players ON Teams.TeamID = Players.TeamID;

commit;

RollBack;
