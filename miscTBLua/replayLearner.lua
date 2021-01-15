-- Prints out the moves made in a replay to a text file
-- By Divine

counter = 0
jointStates = {}
changes = {}
matchFrames = {}
jointList = {"neck", "chest", "lumbar", "abs", "right pec", "right shoulder", "right elbow", "left pec", "left shoulder", "left elbow", "right wrist", "left wrist", "right glute", "left glute", "right hip", "left hip", "right knee", "left knee", "right ankle", "left ankle" }
stateList = {"extended/right rotated/bent right", "contracted/left rotated/bent left", "held", "relaxed"}

local function setVars()
    freeze_game()
    ws = get_world_state()
    unfreeze_game()
    if ws.match_frame == 0 then
        freeze_game()
        --echo("Match frame (beginning) = ".. ws.match_frame)
        --echo("Setting vars")
        for i = 0, 19 do
            jointStates[i] = nil
        end
        for i in pairs(changes) do
            changes[i] = nil
        end
        for i in pairs(matchFrames) do
            matchFrames[i] = nil
        end
        counter = 0
        --echo("Vars set")
        unfreeze_game()
    end
end

add_hook("enter_frame", "setVars", setVars)


local function checkChanges()
    --echo("Beginning to check for changes")
    checker = 0
    ws = get_world_state()
    --echo("Entering loop")
    changes[counter] = {}
    for i = 0, 19 do
        js = get_joint_info(0, i)
        if ((js["state"] ~= jointStates[i]) or (jointStates[i] == nil)) then
            changes[counter][i] = js["state"]
            checker = 1
            --echo("Joint "..i.." Changed")
            --echo("Checker = "..checker)
        end
        jointStates[i] = js["state"]
    end
    --echo("Out of loop")
    if checker == 1 then
        freeze_game()
        unfreeze_game()
        --echo("if checker == 1 passed")
        matchFrames[counter] = ws.match_frame
        --echo("Match frame = "..matchFrames[counter])
        --echo("Counter = "..counter)
        counter = counter + 1
    end
end

add_hook("enter_frame", "check changes", checkChanges)

local function printChanges()
    freeze_game()
    --bouts = get_bouts()
    ws = get_world_state()
    --echo("Match frame end = "..ws.match_frame)
    unfreeze_game()
    if ws.game_frame - ws.match_frame == (1 or 0) then
        --decounter = 0
        freeze_game()
        info = get_player_info(0)
        --echo("Beginning printing")
        --echo(counter)
        randNum = math.random(1000)
        fileName = ""..info.name..""..ws.game_frame.."_"..randNum..".txt"
        --echo(fileName)
        file = io.open(fileName, "w")
        io.output(file)
        --echo("Entering printing loop")
        for i = 0, counter-1 do
            --echo("Entered printing loop")
            io.write("After "..matchFrames[i].." frames, the player changed ")
            --echo("Entering printing loop pt. 2")
            for j = 0, 19 do
                if changes[i][j] ~= nil then
                    io.write(jointList[j+1] .. " to " .. stateList[changes[i][j]] .. ", ")
                end
            end
            --echo("Decounter = "..decounter)
            --decounter = decounter + 1
            io.write("\n")
            io.write("\n")
        end
        --echo("Loop done")
        io.close(file)
        echo(fileName.. " has been saved to your scripts folder")
    end
    io.close(file)
    file:close()
end

add_hook("enter_frame", "print changes", printChanges)
