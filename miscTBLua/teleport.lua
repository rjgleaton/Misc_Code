local function getRotMatrix(key)
    if key == 110 then
        rotMatrix = get_body_info(0,10)
        echo ("Rotation Matrix:")
        echo (rotMatrix.rot.r0 .. " " .. rotMatrix.rot.r1 .. " " .. rotMatrix.rot.r2 .. " " .. rotMatrix.rot.r3)
        echo (rotMatrix.rot.r4 .. " " .. rotMatrix.rot.r5 .. " " .. rotMatrix.rot.r6 .. " " .. rotMatrix.rot.r7)
        echo (rotMatrix.rot.r8 .. " " .. rotMatrix.rot.r9 .. " " .. rotMatrix.rot.r10 .. " " .. rotMatrix.rot.r11)
        echo (rotMatrix.rot.r12 .. " " .. rotMatrix.rot.r13 .. " " .. rotMatrix.rot.r14 .. " " .. rotMatrix.rot.r15)
        echo ("")
        echo ("Z Rotation in degrees: ")
        echo (math.deg(math.acos(rotMatrix.rot.r0)))
        echo ("X Rotation in degrees: ")
        echo (math.deg(math.asin(rotMatrix.rot.r6)))
    end
end

add_hook("key_down", "get_rot_matrix", getRotMatrix)

local function rotate(key)
    if key == 113 then
        aX = {} -- Creates 6 arrays to store x, y, and z position and rotation values
        aY = {}
        aZ = {}
        rX = {}
        --rY = {} never used as Y should never have any rotation
        rZ = {}
        for i = 1, 20 do -- Loop goes through all 20 joints (except head) getting their pos and rot
            bodyPos = get_body_info(0, i)
            aX[i] = (bodyPos.pos.x) -- Fills arrays with x, y, and z positions for each body part
            aY[i] = (bodyPos.pos.y)
            aZ[i] = (bodyPos.pos.z)
            rZ[i] = math.acos(bodyPos.rot.r0)
            rX[i] = math.asin(bodyPos.rot.r6)
            echo("Z rotation of join " .. i .. " before move: ".. rZ[i])
            set_body_rotation(0,i,rX[i],0,rZ[i]+0.174533)
            echo("Z rotation of joint " .. i .. "  after move: ".. rZ[i]+0.174533)
        end
    end
end


add_hook("key_down", "rotate", rotate)
