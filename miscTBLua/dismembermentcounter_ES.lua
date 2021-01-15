-- Divine
-- 9/8/2019
-- For counting dismemberments in a replay (Only counts Uke)
-- ES Version: Also checks for opener challenge frauds

local function countDismemberments()
    ws = get_world_state();
    if(ws.game_frame == ws.match_frame) then -- Check if it's the end of the replay
        count = 0;
        for i = 0, 19 do -- Go through every joint and count the number of DMs
            if get_joint_dismember(1, i) == true then
                count = count + 1
            end
        end
        echo("Number of dismemberments = " .. count) -- Print the number of DMS
    end
end

add_hook("draw3d", "count_dismember", countDismemberments) -- Execute countDismemberments()


local function checkFailure() -- Checks if someone edits uke
    ws2 = get_world_state()
    if ws2.match_frame == 1 then -- Checks if replay is restarted, prevents chat spam bug
        echo("Beginning checks for uke movement")
        timesSaid = 0
    end
    for i = 0, 19 do -- Goes through every joint checking state and DM
        if get_joint_dismember(1, i) == true then -- Checks if joint is DMed
            for j = 0, 19 do -- If joint is DMed it then checks to make sure all joints are relaxed
                jointState = get_joint_info(1,j)
                if jointState["state"] ~= 4 then -- If joint isnt relaxed
                    freeze_game() -- Freeze game
                    if timesSaid == 0 then --Prints message, if statement prevents chat spam
                        echo("Error 1: Challenge failed, uke moved at frame ".. (ws2.game_frame - ws2.match_frame))
                        timesSaid = timesSaid + 1
                    end
                    break
                end
            end
            break
        end
        if get_joint_dismember(1, i) == false then -- If no DM is present
            for j = 0, 19 do -- Goes through making sure each joint is held
                jointState = get_joint_info(1,j)
                if jointState["state"] ~= 3 then -- If joint isn't held
                    local ignore = false; -- Prevents false alarms if the joint checked is relax before DM is found
                    for k = 0, 19 do -- Goes through joints checking for DMs
                        if get_joint_dismember(1,k) == true then
                            ignore = true; -- If it's found and a joint is actually DMed then it's considered a false alarm
                            break
                        end
                    end
                    if(ignore == false) then -- If it's not a false alarm and joint is moved
                        freeze_game()
                        if timesSaid == 0 then -- Print error
                            echo("Error 2: Challenge failed, uke moved at frame ".. (ws2.game_frame - ws2.match_frame))
                            timesSaid = timesSaid + 1
                        end
                    end
                    break
                end
            end
        end
    end
end


add_hook("draw2d", "check_failure", checkFailure) -- Checks for failures
